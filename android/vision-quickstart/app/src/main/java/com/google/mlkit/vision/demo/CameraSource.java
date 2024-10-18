/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Parameters;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.WindowManager;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;
import com.google.android.gms.common.images.Size;
import com.google.mlkit.vision.demo.preference.PreferenceUtils;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Manages the camera and allows UI updates on top of it (e.g. overlaying extra Graphics or
 * displaying extra information). This receives preview frames from the camera at a specified rate,
 * sending those frames to child classes' detectors / classifiers as fast as it is able to process.
 */
public class CameraSource {
  // 카메라가 후면을 사용하는지 나타내는 상수
  @SuppressLint("InlinedApi")
  public static final int CAMERA_FACING_BACK = CameraInfo.CAMERA_FACING_BACK;

  // 카메라가 전면을 사용하는지 나타내는 상수
  @SuppressLint("InlinedApi")
  public static final int CAMERA_FACING_FRONT = CameraInfo.CAMERA_FACING_FRONT;

  // 카메라 이미지 형식 정의 (NV21은 기본적으로 카메라 미리보기에 사용되는 이미지 형식)
  public static final int IMAGE_FORMAT = ImageFormat.NV21;

  // 기본 요청된 카메라 미리보기 너비와 높이 설정
  public static final int DEFAULT_REQUESTED_CAMERA_PREVIEW_WIDTH = 480;
  public static final int DEFAULT_REQUESTED_CAMERA_PREVIEW_HEIGHT = 360;

  // 로그 메시지에 사용될 태그 정의
  private static final String TAG = "MIDemoApp:CameraSource";

  /**
   * The dummy surface texture must be assigned a chosen name. Since we never use an OpenGL context,
   * we can choose any ID we want here. The dummy surface texture is not a crazy hack - it is
   * actually how the camera team recommends using the camera without a preview.
   *
   * dummySurfaceTexture는 카메라의 미리보기를 표시하지 않고도 카메라를 작동시키기 위해 사용됩니다. 카메라 팀에서는 이 방식을 권장합니다. OpenGL과 상관없이 고유한 ID를 지정할 수 있습니다.
   */
  //미리보기 없이 카메라를 사용하는 경우 더미 SurfaceTexture에 사용할 ID 값
  private static final int DUMMY_TEXTURE_NAME = 100;

  /**
   * If the absolute difference between a preview size aspect ratio and a picture size aspect ratio
   * is less than this tolerance, they are considered to be the same aspect ratio.
   *
   * 카메라의 미리보기 크기와 사진 크기가 동일한 비율을 유지해야 하는데, 두 크기 간의 비율 차이가 0.01보다 작으면 같은 비율로 처리됩니다.
   */
  private static final float ASPECT_RATIO_TOLERANCE = 0.01f;

  // Activity에 대한 참조 저장
  protected Activity activity;

  // 카메라 객체에 대한 참조 저장
  private Camera camera;

  // 기본 카메라 방향 설정 (후면 카메라 사용)
  private int facing = CAMERA_FACING_BACK;

  /** Rotation of the device, and thus the associated preview images captured from the device.
   * 카메라와 기본 감지기의 리소스를 해제하고 카메라를 중지시킵니다.
   * */
  private int rotationDegrees;

  private Size previewSize;

  // 카메라의 요청된 프레임 속도(FPS)
  private static final float REQUESTED_FPS = 30.0f;

  // 자동 초점 사용 여부
  private static final boolean REQUESTED_AUTO_FOCUS = true;

  // This instance needs to be held onto to avoid GC of its underlying resources. Even though it
  // isn't used outside of the method that creates it, it still must have hard references maintained
  // to it.
  // 더미 SurfaceTexture에 대한 참조 저장
  private SurfaceTexture dummySurfaceTexture;

  // 그래픽 오버레이 참조 저장 (카메라 미리보기 위에 그리기 위한 레이어)
  private final GraphicOverlay graphicOverlay;

  /**
   * Dedicated thread and associated runnable for calling into the detector with frames, as the
   * frames become available from the camera.
   * 카메라에서 수신한 프레임을 처리하는 processingRunnable을 실행하는 스레드를 나타냅니다. 이 스레드는 프레임이 감지기로 전달될 때 실행됩니다.
   */
  private Thread processingThread;

  //프레임을 처리하는 Runnable 객체입니다. 카메라에서 수신된 프레임을 감지기로 전달하고, 프레임 처리를 위한 코드를 실행하는 역할을 합니다.
  private final FrameProcessingRunnable processingRunnable;
  //스레드 간 동기화를 위한 락 오브젝트입니다. 여러 스레드가 processingRunnable을 동시에 접근하지 않도록 동기화를 위한 락으로 사용됩니다.
  private final Object processorLock = new Object();
  //카메라에서 수신한 프레임을 처리하는 감지기(프레임 프로세서)를 나타냅니다. 예를 들어, 이미지에서 특정 객체를 감지하거나 분석하는 작업을 수행하는 프로세서입니다.
  private VisionImageProcessor frameProcessor;

  /**
   * Map to convert between a byte array, received from the camera, and its associated byte buffer.
   * We use byte buffers internally because this is a more efficient way to call into native code
   * later (avoids a potential copy).
   *
   * <p><b>Note:</b> uses IdentityHashMap here instead of HashMap because the behavior of an array's
   * equals, hashCode and toString methods is both useless and unexpected. IdentityHashMap enforces
   * identity ('==') check on the keys.
   */

  /**
   * 카메라에서 받은 바이트 배열과 해당하는 바이트 버퍼를 변환하는 맵입니다.
   * 나중에 네이티브 코드로 호출할 때 더 효율적으로 처리하기 위해 내부적으로 바이트 버퍼를 사용합니다
   * (불필요한 복사를 피할 수 있습니다).
   *
   * <p><b>참고:</b> 여기서는 HashMap 대신 IdentityHashMap을 사용합니다. 배열의 equals, hashCode, toString 메소드의
   * 동작이 쓸모없고 예상치 못한 결과를 가져올 수 있기 때문입니다. IdentityHashMap은 키에 대해 객체 참조 비교('==')를 강제합니다.
   */
  private final IdentityHashMap<byte[], ByteBuffer> bytesToByteBuffer = new IdentityHashMap<>();

  // 클래스 생성자 객체가 만들 때 호출도미. 초기화 작업을 수행함.
  public CameraSource(Activity activity, GraphicOverlay overlay) {
    this.activity = activity;
    graphicOverlay = overlay;
    graphicOverlay.clear();
    processingRunnable = new FrameProcessingRunnable();
  }

  // ==============================================================================================
  // Public
  // ==============================================================================================

  /** Stops the camera and releases the resources of the camera and underlying detector.
   * 카메라를 중지하고 카메라 및 하위 감지기의 리소스를 해제합니다.
   * */
  public void release() {
    synchronized (processorLock) {
      stop();
      cleanScreen();

      if (frameProcessor != null) {
        frameProcessor.stop();
      }
    }
  }

  /**
   * Opens the camera and starts sending preview frames to the underlying detector. The preview
   * frames are not displayed.
   *
   * @throws IOException if the camera's preview texture or display could not be initialized
   *
   *
   * 카메라를 열고 기본 감지기로 미리보기 프레임을 전송하기 시작합니다.
   * 미리보기 프레임은 화면에 표시되지 않습니다.
   *
   *@throws IOException 카메라의 미리보기 텍스처 또는 디스플레이를 초기화할 수 없는 경우 발생
   *
   */
  @RequiresPermission(Manifest.permission.CAMERA)
  public synchronized CameraSource start() throws IOException {
    if (camera != null) {
      return this;
    }

    camera = createCamera();
    dummySurfaceTexture = new SurfaceTexture(DUMMY_TEXTURE_NAME);
    camera.setPreviewTexture(dummySurfaceTexture);
    camera.startPreview();

    processingThread = new Thread(processingRunnable);
    processingRunnable.setActive(true);
    processingThread.start();
    return this;
  }

  /**
   * Opens the camera and starts sending preview frames to the underlying detector. The supplied
   * surface holder is used for the preview so frames can be displayed to the user.
   *
   * @param surfaceHolder the surface holder to use for the preview frames
   * @throws IOException if the supplied surface holder could not be used as the preview display
   */

  /**
   * 카메라를 열고 기본 감지기로 미리보기 프레임을 전송하기 시작합니다. 제공된 서피스 홀더가 미리보기에 사용되어
   * 프레임이 사용자에게 표시됩니다.
   *
   * @param surfaceHolder 미리보기 프레임을 표시할 때 사용할 서피스 홀더
   * @throws IOException 제공된 서피스 홀더를 미리보기 디스플레이로 사용할 수 없는 경우 발생
   */
  @RequiresPermission(Manifest.permission.CAMERA)
  public synchronized CameraSource start(SurfaceHolder surfaceHolder) throws IOException {
    if (camera != null) {
      return this;
    }

    camera = createCamera();
    camera.setPreviewDisplay(surfaceHolder);
    camera.startPreview();

    processingThread = new Thread(processingRunnable);
    processingRunnable.setActive(true);
    processingThread.start();
    return this;
  }

  /**
   * Closes the camera and stops sending frames to the underlying frame detector.
   *
   * <p>This camera source may be restarted again by calling {@link #start()} or {@link
   * #start(SurfaceHolder)}.
   *
   * <p>Call {@link #release()} instead to completely shut down this camera source and release the
   * resources of the underlying detector.
   */
  /**
   * 카메라를 닫고 기본 프레임 감지기로의 프레임 전송을 중지합니다.
   *
   * <p>이 카메라 소스는 {@link #start()} 또는 {@link #start(SurfaceHolder)}를 호출하여 다시 시작할 수 있습니다.
   *
   * <p>이 카메라 소스를 완전히 종료하고 기본 감지기의 리소스를 해제하려면 {@link #release()}를 호출하십시오.
   */
  public synchronized void stop() {
    processingRunnable.setActive(false);
    if (processingThread != null) {
      try {
        // Wait for the thread to complete to ensure that we can't have multiple threads
        // executing at the same time (i.e., which would happen if we called start too
        // quickly after stop).
        // 스레드가 완료될 때까지 기다려서 동시에 여러 스레드가 실행되지 않도록 합니다
        // (예: stop 후에 start를 너무 빨리 호출하면 이러한 문제가 발생할 수 있습니다).
        processingThread.join();
      } catch (InterruptedException e) {
        Log.d(TAG, "Frame processing thread interrupted on release.");
      }
      processingThread = null;
    }

    if (camera != null) {
      camera.stopPreview();
      camera.setPreviewCallbackWithBuffer(null);
      try {
        camera.setPreviewTexture(null);
        dummySurfaceTexture = null;
        camera.setPreviewDisplay(null);
      } catch (Exception e) {
        Log.e(TAG, "Failed to clear camera preview: " + e);
      }
      camera.release();
      camera = null;
    }

    // Release the reference to any image buffers, since these will no longer be in use.
    bytesToByteBuffer.clear();
  }

  /** Changes the facing of the camera. */
  ///** 카메라의 방향을 변경합니다. */
  //synchronized 키워드를 사용하여 이 메소드가 여러 스레드에서 동시에 호출되지 않도록 동기화합니다.
  public synchronized void setFacing(int facing) {
    if ((facing != CAMERA_FACING_BACK) && (facing != CAMERA_FACING_FRONT)) {
      throw new IllegalArgumentException("Invalid camera: " + facing);
    }
    this.facing = facing;
  }

  /** Returns the preview size that is currently in use by the underlying camera. */
  /** 현재 기본 카메라에서 사용 중인 미리보기 크기를 반환합니다. */
  public Size getPreviewSize() {
    return previewSize;
  }

  /**
   * Returns the selected camera; one of {@link #CAMERA_FACING_BACK} or {@link
   * #CAMERA_FACING_FRONT}.
   */
  /**
   * 선택된 카메라를 반환합니다; {@link #CAMERA_FACING_BACK} 또는 {@link #CAMERA_FACING_FRONT} 중 하나.
   */
  public int getCameraFacing() {
    return facing;
  }

  public boolean setZoom(float zoomRatio) {
    Log.d(TAG, "setZoom: " + zoomRatio);
    if (camera == null) {
      return false;
    }

    Parameters parameters = camera.getParameters();
    parameters.setZoom(getZoomValue(parameters, zoomRatio));
    camera.setParameters(parameters);
    return true;
  }

  /**
   * Calculate the zoom value of the target zoom ratio.
   *
   * <p>According to the camera API, {@link Parameters#getZoomRatios()} will return a list of zoom
   * ratios with length {@link Parameters#getMaxZoom()}+1. Each of this value indicates a actual
   * zoom ratio of the camera.
   *
   * <p>E.g. Assume {@link Parameters#getZoomRatios()} return {@code [100, 114, 131, 151, 174, 200,
   * 234, 268, 300]}, where {@link Parameters#getMaxZoom()}=8. It means, {@code setZoom(0)} will
   * actual perform 1.00x to the camera, {@code setZoom(1)} will actual perform 1.14x to the camera,
   * {@code setZoom(2)} will actual perform 1.31x to the camera, ..., {@code setZoom(8)} will actual
   * perform 3.00x to the camera.
   *
   * @param params The parameters of the camera.
   * @param zoomRatio The target zoom ratio.
   * @return The maximum zoom value that will not exceed the target {@code zoomRatio}.
   */
  /**
   * 대상 줌 비율에 대한 줌 값을 계산합니다.
   *
   * <p>카메라 API에 따르면, {@link Parameters#getZoomRatios()}는 길이가 {@link Parameters#getMaxZoom()}+1인
   * 줌 비율 목록을 반환합니다. 이 목록의 각 값은 카메라의 실제 줌 비율을 나타냅니다.
   *
   * <p>예시: {@link Parameters#getZoomRatios()}가 {@code [100, 114, 131, 151, 174, 200, 234, 268, 300]}를 반환하고,
   * {@link Parameters#getMaxZoom()} 값이 8인 경우, {@code setZoom(0)}은 카메라에 대해 1.00배 줌을 수행하고,
   * {@code setZoom(1)}은 1.14배 줌을 수행하며, {@code setZoom(2)}는 1.31배 줌을 수행하고,
   * ..., {@code setZoom(8)}은 카메라에서 3.00배 줌을 수행하게 됩니다.
   *
   * @param params 카메라의 파라미터.
   * @param zoomRatio 대상 줌 비율.
   * @return 대상 {@code zoomRatio}를 초과하지 않는 최대 줌 값.
   */

  private static int getZoomValue(Camera.Parameters params, float zoomRatio) {
    int zoom = (int) (Math.max(zoomRatio, 1) * 100);
    List<Integer> zoomRatios = params.getZoomRatios();
    int maxZoom = params.getMaxZoom();
    for (int i = 0; i < maxZoom; ++i) {
      if (zoomRatios.get(i + 1) > zoom) {
        return i;
      }
    }
    return maxZoom;
  }

  /**
   * Opens the camera and applies the user settings.
   *
   * @throws IOException if camera cannot be found or preview cannot be processed
   */
  /**
   * 카메라를 열고 사용자 설정을 적용합니다.
   *
   * @throws IOException 카메라를 찾을 수 없거나 미리보기를 처리할 수 없는 경우 발생
   */
  @SuppressLint("InlinedApi") // Lint 경고를 억제하기 위한 어노테이션입니다. API 레벨 관련 경고를 무시합니다.
  private Camera createCamera() throws IOException {
    int requestedCameraId = getIdForRequestedCamera(facing); // 요청된 카메라 방향(전면 또는 후면)에 맞는 카메라 ID를 가져옵니다.
    if (requestedCameraId == -1) { // 카메라 ID를 찾지 못한 경우, IOException을 던집니다. -1은 해당 방향의 카메라가 없다는 뜻입니다.
      throw new IOException("Could not find requested camera.");
    }
    Camera camera = Camera.open(requestedCameraId); // 카메라 ID로 카메라를 엽니다. 카메라 객체를 반환합니다.

    // 카메라 미리보기와 사진 크기 쌍을 가져옵니다. 설정에 따라 카메라의 최적 크기를 선택합니다.
    SizePair sizePair = PreferenceUtils.getCameraPreviewSizePair(activity, requestedCameraId);

    // 카메라 설정에 맞는 미리보기 크기를 찾지 못한 경우, 기본 미리보기 크기와 사진 크기 쌍을 선택합니다.
    if (sizePair == null) {
      sizePair =
          selectSizePair(
              camera,
              DEFAULT_REQUESTED_CAMERA_PREVIEW_WIDTH,
              DEFAULT_REQUESTED_CAMERA_PREVIEW_HEIGHT);
    }

    if (sizePair == null) { // 적절한 미리보기 크기를 찾지 못하면 IOException을 던집니다.
      throw new IOException("Could not find suitable preview size.");
    }

    previewSize = sizePair.preview; // 선택된 미리보기 크기를 저장합니다.
    Log.v(TAG, "Camera preview size: " + previewSize);

    // 요청된 프레임 속도(FPS)에 맞는 카메라 프레임 속도 범위를 선택합니다.
    int[] previewFpsRange = selectPreviewFpsRange(camera, REQUESTED_FPS);
    if (previewFpsRange == null) {
      throw new IOException("Could not find suitable preview frames per second range.");
    }

    // 카메라의 파라미터를 가져옵니다. 이를 통해 카메라 설정을 변경할 수 있습니다.
    Camera.Parameters parameters = camera.getParameters();

    // 사진 크기를 가져옵니다. 사진 크기는 미리보기 크기와 비율이 일치해야 합니다.
    Size pictureSize = sizePair.picture;
    if (pictureSize != null) { // 사진 크기가 존재하면 해당 크기를 로그로 출력하고, 카메라 파라미터에 설정합니다.
      Log.v(TAG, "Camera picture size: " + pictureSize);
      parameters.setPictureSize(pictureSize.getWidth(), pictureSize.getHeight());
    }
    // 선택된 미리보기 크기를 카메라 파라미터에 설정합니다.
    parameters.setPreviewSize(previewSize.getWidth(), previewSize.getHeight());
    parameters.setPreviewFpsRange( // 선택된 프레임 속도 범위를 카메라 파라미터에 설정합니다.
        previewFpsRange[Camera.Parameters.PREVIEW_FPS_MIN_INDEX],
        previewFpsRange[Camera.Parameters.PREVIEW_FPS_MAX_INDEX]);

    // Use YV12 so that we can exercise YV12->NV21 auto-conversion logic for OCR detection
    // 이미지 형식을 설정합니다. 이 경우 NV21 형식으로 설정하여 OCR 등에서 사용될 수 있도록 합니다.
    parameters.setPreviewFormat(IMAGE_FORMAT);

    // 카메라의 회전 방향을 설정합니다. 이를 통해 장치의 회전과 카메라의 방향을 일치시킵니다.
    setRotation(camera, parameters, requestedCameraId);

    // 자동 초점 기능을 요청한 경우, 카메라가 자동 초점 모드를 지원하는지 확인하고, 지원하면 설정합니다. 지원하지 않으면 로그에 이를 출력합니다.
    if (REQUESTED_AUTO_FOCUS) {
      if (parameters
          .getSupportedFocusModes()
          .contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
        parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
      } else {
        Log.i(TAG, "Camera auto focus is not supported on this device.");
      }
    }

    // 설정된 카메라 파라미터를 카메라에 적용합니다.
    camera.setParameters(parameters);

    // Four frame buffers are needed for working with the camera:
    //
    //   one for the frame that is currently being executed upon in doing detection
    //   one for the next pending frame to process immediately upon completing detection
    //   two for the frames that the camera uses to populate future preview images
    //
    // Through trial and error it appears that two free buffers, in addition to the two buffers
    // used in this code, are needed for the camera to work properly.  Perhaps the camera has
    // one thread for acquiring images, and another thread for calling into user code.  If only
    // three buffers are used, then the camera will spew thousands of warning messages when
    // detection takes a non-trivial amount of time.

    // 카메라에서 작동하기 위해 4개의 프레임 버퍼가 필요합니다:
    //
    //   - 하나는 감지 작업을 실행 중인 프레임을 위한 버퍼
    //   - 하나는 감지가 완료된 후 즉시 처리할 대기 중인 다음 프레임을 위한 버퍼
    //   - 두 개는 카메라가 미래의 미리보기 이미지를 채우기 위해 사용하는 버퍼
    //
    // 시행착오 끝에, 이 코드에서 사용되는 두 개의 버퍼 외에 두 개의 추가적인 버퍼가
    // 카메라가 제대로 작동하기 위해 필요하다는 것을 알게 되었습니다. 아마도 카메라는
    // 이미지를 수집하는 하나의 스레드와 사용자 코드를 호출하는 또 다른 스레드를 갖고 있을 것입니다.
    // 만약 버퍼가 세 개만 사용되면, 감지 시간이 길어질 때 카메라에서 수천 개의 경고 메시지가 발생할 수 있습니다.

    // 카메라에 4개의 버퍼를 추가합니다. 카메라는 프레임을 처리할 때 4개의 버퍼를 사용하여 최적화된 프레임 처리를 수행합니다.
    // 이를 통해 프레임을 감지하는 동안 추가적인 프레임을 수집하여 처리 지연을 방지합니다.
    camera.setPreviewCallbackWithBuffer(new CameraPreviewCallback());
    camera.addCallbackBuffer(createPreviewBuffer(previewSize));
    camera.addCallbackBuffer(createPreviewBuffer(previewSize));
    camera.addCallbackBuffer(createPreviewBuffer(previewSize));
    camera.addCallbackBuffer(createPreviewBuffer(previewSize));

    return camera;
  }

  /**
   * Gets the id for the camera specified by the direction it is facing. Returns -1 if no such
   * camera was found.
   *
   * @param facing the desired camera (front-facing or rear-facing)
   */

  /**
   * 지정된 방향을 기준으로 카메라의 ID를 가져옵니다. 해당 방향의 카메라를 찾지 못한 경우 -1을 반환합니다.
   *
   * @param facing 원하는 카메라 방향 (전면 카메라 또는 후면 카메라)
   */
  private static int getIdForRequestedCamera(int facing) {
    CameraInfo cameraInfo = new CameraInfo(); //카메라의 정보를 담기 위한 CameraInfo 객체를 생성합니다.

    /*이 반복문은 장치에 연결된 모든 카메라의 정보를 가져옵니다.
    Camera.getNumberOfCameras()를 사용하여 장치에 연결된
    카메라의 개수를 얻고, 각 카메라에 대해 Camera.getCameraInfo(i, cameraInfo)를 호출하여
    해당 카메라의 정보를 cameraInfo 객체에 저장합니다.*/
    for (int i = 0; i < Camera.getNumberOfCameras(); ++i) {
      Camera.getCameraInfo(i, cameraInfo);
      if (cameraInfo.facing == facing) {
        /*그 카메라가 원하는 방향(facing)과 일치하는지 확인합니다. 일치하면 해당 카메라의 ID인 i를 반환합니다.*/
        return i;
      }
    }
    return -1;
  }

  /**
   * Selects the most suitable preview and picture size, given the desired width and height.
   *
   * <p>Even though we only need to find the preview size, it's necessary to find both the preview
   * size and the picture size of the camera together, because these need to have the same aspect
   * ratio. On some hardware, if you would only set the preview size, you will get a distorted
   * image.
   *
   * @param camera the camera to select a preview size from
   * @param desiredWidth the desired width of the camera preview frames
   * @param desiredHeight the desired height of the camera preview frames
   * @return the selected preview and picture size pair
   */
  /**
   * 주어진 너비와 높이를 기반으로 가장 적합한 미리보기 및 사진 크기를 선택합니다.
   *
   * <p>비록 미리보기 크기만 찾으면 되지만, 미리보기 크기와 사진 크기를 함께 찾아야 합니다.
   * 이는 두 크기가 동일한 비율(aspect ratio)을 가져야 하기 때문입니다. 일부 하드웨어에서는
   * 미리보기 크기만 설정하면 이미지가 왜곡될 수 있습니다.
   *
   * @param camera 미리보기 크기를 선택할 카메라 객체
   * @param desiredWidth 카메라 미리보기 프레임의 원하는 너비
   * @param desiredHeight 카메라 미리보기 프레임의 원하는 높이
   * @return 선택된 미리보기와 사진 크기 쌍
   */
  //이 주석은 메소드가 주어진 너비와 높이에 따라 카메라의 미리보기 및 사진 크기 중
  // 가장 적합한 크기를 선택하는 것을 설명합니다. 미리보기 크기만 설정할 경우 이미지가
  // 왜곡될 수 있으므로, 미리보기와 사진 크기가 동일한 비율을 가져야 합니다.
  public static SizePair selectSizePair(Camera camera, int desiredWidth, int desiredHeight) {
    List<SizePair> validPreviewSizes = generateValidPreviewSizeList(camera);

    // The method for selecting the best size is to minimize the sum of the differences between
    // the desired values and the actual values for width and height.  This is certainly not the
    // only way to select the best size, but it provides a decent tradeoff between using the
    // closest aspect ratio vs. using the closest pixel area.
    // 최적의 크기를 선택하는 방법은 원하는 값과 실제 값(너비와 높이)의 차이의 합을 최소화하는 것입니다.
    // 이것이 최적의 크기를 선택하는 유일한 방법은 아니지만,
    // 가장 가까운 비율(aspect ratio)과 가장 가까운 픽셀 영역(pixel area) 간의 적절한 절충안을 제공합니다.
    SizePair selectedPair = null;
    int minDiff = Integer.MAX_VALUE;
    for (SizePair sizePair : validPreviewSizes) {
      Size size = sizePair.preview;
      int diff =
          Math.abs(size.getWidth() - desiredWidth) + Math.abs(size.getHeight() - desiredHeight);
      if (diff < minDiff) {
        selectedPair = sizePair;
        minDiff = diff;
      }
    }

    return selectedPair;
  }

  /**
   * Stores a preview size and a corresponding same-aspect-ratio picture size. To avoid distorted
   * preview images on some devices, the picture size must be set to a size that is the same aspect
   * ratio as the preview size or the preview may end up being distorted. If the picture size is
   * null, then there is no picture size with the same aspect ratio as the preview size.
   */
  /**
   * 미리보기 크기와 동일한 비율을 가진 사진 크기를 저장합니다. 일부 장치에서 미리보기 이미지가 왜곡되는 것을 방지하기 위해,
   * 사진 크기는 미리보기 크기와 동일한 비율(aspect ratio)로 설정되어야 합니다. 만약 사진 크기가 null인 경우,
   * 미리보기 크기와 동일한 비율의 사진 크기가 없다는 의미입니다.
   */
  public static class SizePair {
    public final Size preview;
    @Nullable public final Size picture;

    SizePair(Camera.Size previewSize, @Nullable Camera.Size pictureSize) {
      preview = new Size(previewSize.width, previewSize.height);
      picture = pictureSize != null ? new Size(pictureSize.width, pictureSize.height) : null;
    }

    public SizePair(Size previewSize, @Nullable Size pictureSize) {
      preview = previewSize;
      picture = pictureSize;
    }
  }

  /**
   * Generates a list of acceptable preview sizes. Preview sizes are not acceptable if there is not
   * a corresponding picture size of the same aspect ratio. If there is a corresponding picture size
   * of the same aspect ratio, the picture size is paired up with the preview size.
   *
   * <p>This is necessary because even if we don't use still pictures, the still picture size must
   * be set to a size that is the same aspect ratio as the preview size we choose. Otherwise, the
   * preview images may be distorted on some devices.
   */
  /**
   * 허용 가능한 미리보기 크기 목록을 생성합니다. 동일한 비율(aspect ratio)을 가진 사진 크기가 없으면
   * 미리보기 크기는 허용되지 않습니다. 동일한 비율을 가진 사진 크기가 있으면, 그 사진 크기가 미리보기 크기와 쌍으로 묶입니다.
   *
   * <p>이는 정지 사진을 사용하지 않더라도, 우리가 선택한 미리보기 크기와 동일한 비율의 사진 크기를 설정해야 하기 때문에 필요합니다.
   * 그렇지 않으면, 일부 장치에서는 미리보기 이미지가 왜곡될 수 있습니다.
   */
  // 메소드는 카메라 객체를 매개변수로 받아, 해당 카메라에서 사용할 수 있는 유효한 미리보기 및 사진 크기 쌍(SizePair)의 목록을 반환합니다.
  public static List<SizePair> generateValidPreviewSizeList(Camera camera) {
    //카메라의 설정 정보를 담고 있는 Camera.Parameters 객체를 가져옵니다.
    // 이를 통해 카메라의 지원 가능한 미리보기 및 사진 크기 등을 가져올 수 있습니다.
    Camera.Parameters parameters = camera.getParameters();

    //카메라에서 지원하는 미리보기 크기 목록을 가져옵니다. 이 목록은 Camera.Size 객체의 리스트로 반환됩니다.
    List<Camera.Size> supportedPreviewSizes = parameters.getSupportedPreviewSizes();
    //카메라에서 지원하는 사진 크기 목록을 가져옵니다. 이 목록도 Camera.Size 객체의 리스트로 반환됩니다.
    List<Camera.Size> supportedPictureSizes = parameters.getSupportedPictureSizes();

    //유효한 미리보기 및 사진 크기 쌍을 저장할 리스트를 초기화합니다.
    // 이 리스트에는 미리보기 크기와 동일한 비율을 가진 사진 크기가 쌍으로 저장됩니다.
    List<SizePair> validPreviewSizes = new ArrayList<>();

    // 지원되는 모든 미리보기 크기를 반복문으로 순회합니다. 각 미리보기 크기의 비율을 계산하여 previewAspectRatio에 저장합니다.
    for (Camera.Size previewSize : supportedPreviewSizes) {
      float previewAspectRatio = (float) previewSize.width / (float) previewSize.height;

      // By looping through the picture sizes in order, we favor the higher resolutions.
      // We choose the highest resolution in order to support taking the full resolution
      // picture later.
      // 사진 크기를 순서대로 반복함으로써, 우리는 더 높은 해상도를 선호합니다.
      // 나중에 전체 해상도 사진을 촬영할 수 있도록 가장 높은 해상도를 선택합니다.

      /*각 미리보기 크기에 대해, 지원되는 모든 사진 크기를 순회하며, 그 비율을 pictureAspectRatio에 계산합니다.
      두 비율의 차이가 허용 오차(ASPECT_RATIO_TOLERANCE) 이내이면,
      해당 미리보기 크기와 사진 크기를 SizePair로 쌍을 이루어 리스트에 추가합니다.
      그런 후, 같은 비율을 가진 사진 크기를 찾으면 반복문을 중단합니다(break).*/
      for (Camera.Size pictureSize : supportedPictureSizes) {
        float pictureAspectRatio = (float) pictureSize.width / (float) pictureSize.height;
        if (Math.abs(previewAspectRatio - pictureAspectRatio) < ASPECT_RATIO_TOLERANCE) {
          validPreviewSizes.add(new SizePair(previewSize, pictureSize));
          break;
        }
      }
    }

    // If there are no picture sizes with the same aspect ratio as any preview sizes, allow all
    // of the preview sizes and hope that the camera can handle it.  Probably unlikely, but we
    // still account for it.
    // 미리보기 크기와 동일한 비율을 가진 사진 크기가 없을 경우, 모든 미리보기 크기를 허용하고
    // 카메라가 이를 처리할 수 있기를 기대합니다. 이는 아마 드문 일이겠지만, 여전히 대비합니다.


    /*만약 비율이 일치하는 미리보기 및 사진 크기를 찾지 못한 경우, 경고 로그를 출력하고,
    미리보기 크기와 null로 구성된 SizePair를 추가합니다.
    이 경우, 사진 크기는 설정하지 않으며 미리보기 크기만 사용하도록 설정합니다.*/
    if (validPreviewSizes.size() == 0) {
      Log.w(TAG, "No preview sizes have a corresponding same-aspect-ratio picture size");
      for (Camera.Size previewSize : supportedPreviewSizes) {
        // The null picture size will let us know that we shouldn't set a picture size.
        validPreviewSizes.add(new SizePair(previewSize, null));
      }
    }

    return validPreviewSizes;
  }

  /**
   * Selects the most suitable preview frames per second range, given the desired frames per second.
   *
   * @param camera the camera to select a frames per second range from
   * @param desiredPreviewFps the desired frames per second for the camera preview frames
   * @return the selected preview frames per second range
   */
  /**
   * 원하는 초당 프레임 수(FPS)에 따라 가장 적합한 미리보기 프레임 속도 범위를 선택합니다.
   *
   * @param camera 프레임 속도 범위를 선택할 카메라 객체
   * @param desiredPreviewFps 카메라 미리보기 프레임을 위한 원하는 초당 프레임 수(FPS)
   * @return 선택된 미리보기 프레임 속도 범위
   */
  @SuppressLint("InlinedApi")
  private static int[] selectPreviewFpsRange(Camera camera, float desiredPreviewFps) {
    // The camera API uses integers scaled by a factor of 1000 instead of floating-point frame
    // rates.
    int desiredPreviewFpsScaled = (int) (desiredPreviewFps * 1000.0f);

    // Selects a range with whose upper bound is as close as possible to the desired fps while its
    // lower bound is as small as possible to properly expose frames in low light conditions. Note
    // that this may select a range that the desired value is outside of. For example, if the
    // desired frame rate is 30.5, the range (30, 30) is probably more desirable than (30, 40).
    int[] selectedFpsRange = null;
    int minUpperBoundDiff = Integer.MAX_VALUE;
    int minLowerBound = Integer.MAX_VALUE;
    List<int[]> previewFpsRangeList = camera.getParameters().getSupportedPreviewFpsRange();
    for (int[] range : previewFpsRangeList) {
      int upperBoundDiff =
          Math.abs(desiredPreviewFpsScaled - range[Camera.Parameters.PREVIEW_FPS_MAX_INDEX]);
      int lowerBound = range[Camera.Parameters.PREVIEW_FPS_MIN_INDEX];
      if (upperBoundDiff <= minUpperBoundDiff && lowerBound <= minLowerBound) {
        selectedFpsRange = range;
        minUpperBoundDiff = upperBoundDiff;
        minLowerBound = lowerBound;
      }
    }
    return selectedFpsRange;
  }

  /**
   * Calculates the correct rotation for the given camera id and sets the rotation in the
   * parameters. It also sets the camera's display orientation and rotation.
   *
   * @param parameters the camera parameters for which to set the rotation
   * @param cameraId the camera id to set rotation based on
   */
  private void setRotation(Camera camera, Camera.Parameters parameters, int cameraId) {
    WindowManager windowManager = (WindowManager) activity.getSystemService(Context.WINDOW_SERVICE);
    int degrees = 0;
    int rotation = windowManager.getDefaultDisplay().getRotation();
    switch (rotation) {
      case Surface.ROTATION_0:
        degrees = 0;
        break;
      case Surface.ROTATION_90:
        degrees = 90;
        break;
      case Surface.ROTATION_180:
        degrees = 180;
        break;
      case Surface.ROTATION_270:
        degrees = 270;
        break;
      default:
        Log.e(TAG, "Bad rotation value: " + rotation);
    }

    CameraInfo cameraInfo = new CameraInfo();
    Camera.getCameraInfo(cameraId, cameraInfo);

    int displayAngle;
    if (cameraInfo.facing == CameraInfo.CAMERA_FACING_FRONT) {
      this.rotationDegrees = (cameraInfo.orientation + degrees) % 360;
      displayAngle = (360 - this.rotationDegrees) % 360; // compensate for it being mirrored
    } else { // back-facing
      this.rotationDegrees = (cameraInfo.orientation - degrees + 360) % 360;
      displayAngle = this.rotationDegrees;
    }
    Log.d(TAG, "Display rotation is: " + rotation);
    Log.d(TAG, "Camera face is: " + cameraInfo.facing);
    Log.d(TAG, "Camera rotation is: " + cameraInfo.orientation);
    // This value should be one of the degrees that ImageMetadata accepts: 0, 90, 180 or 270.
    Log.d(TAG, "RotationDegrees is: " + this.rotationDegrees);

    camera.setDisplayOrientation(displayAngle);
    parameters.setRotation(this.rotationDegrees);
  }

  /**
   * Creates one buffer for the camera preview callback. The size of the buffer is based off of the
   * camera preview size and the format of the camera image.
   *
   * @return a new preview buffer of the appropriate size for the current camera settings
   */
  @SuppressLint("InlinedApi")
  private byte[] createPreviewBuffer(Size previewSize) {
    int bitsPerPixel = ImageFormat.getBitsPerPixel(IMAGE_FORMAT);
    long sizeInBits = (long) previewSize.getHeight() * previewSize.getWidth() * bitsPerPixel;
    int bufferSize = (int) Math.ceil(sizeInBits / 8.0d) + 1;

    // Creating the byte array this way and wrapping it, as opposed to using .allocate(),
    // should guarantee that there will be an array to work with.
    byte[] byteArray = new byte[bufferSize];
    ByteBuffer buffer = ByteBuffer.wrap(byteArray);
    if (!buffer.hasArray() || (buffer.array() != byteArray)) {
      // I don't think that this will ever happen.  But if it does, then we wouldn't be
      // passing the preview content to the underlying detector later.
      throw new IllegalStateException("Failed to create valid buffer for camera source.");
    }

    bytesToByteBuffer.put(byteArray, buffer);
    return byteArray;
  }

  // ==============================================================================================
  // Frame processing
  // ==============================================================================================

  /** Called when the camera has a new preview frame. */
  private class CameraPreviewCallback implements Camera.PreviewCallback {
    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
      processingRunnable.setNextFrame(data, camera);
    }
  }

  public void setMachineLearningFrameProcessor(VisionImageProcessor processor) {
    synchronized (processorLock) {
      cleanScreen();
      if (frameProcessor != null) {
        frameProcessor.stop();
      }
      frameProcessor = processor;
    }
  }

  /**
   * This runnable controls access to the underlying receiver, calling it to process frames when
   * available from the camera. This is designed to run detection on frames as fast as possible
   * (i.e., without unnecessary context switching or waiting on the next frame).
   *
   * <p>While detection is running on a frame, new frames may be received from the camera. As these
   * frames come in, the most recent frame is held onto as pending. As soon as detection and its
   * associated processing is done for the previous frame, detection on the mostly recently received
   * frame will immediately start on the same thread.
   */
  private class FrameProcessingRunnable implements Runnable {

    // This lock guards all of the member variables below.
    private final Object lock = new Object();
    private boolean active = true;

    // These pending variables hold the state associated with the new frame awaiting processing.
    private ByteBuffer pendingFrameData;

    FrameProcessingRunnable() {}

    /** Marks the runnable as active/not active. Signals any blocked threads to continue. */
    void setActive(boolean active) {
      synchronized (lock) {
        this.active = active;
        lock.notifyAll();
      }
    }

    /**
     * Sets the frame data received from the camera. This adds the previous unused frame buffer (if
     * present) back to the camera, and keeps a pending reference to the frame data for future use.
     */
    @SuppressWarnings("ByteBufferBackingArray")
    void setNextFrame(byte[] data, Camera camera) {
      synchronized (lock) {
        if (pendingFrameData != null) {
          camera.addCallbackBuffer(pendingFrameData.array());
          pendingFrameData = null;
        }

        if (!bytesToByteBuffer.containsKey(data)) {
          Log.d(
              TAG,
              "Skipping frame. Could not find ByteBuffer associated with the image "
                  + "data from the camera.");
          return;
        }

        pendingFrameData = bytesToByteBuffer.get(data);

        // Notify the processor thread if it is waiting on the next frame (see below).
        lock.notifyAll();
      }
    }

    /**
     * As long as the processing thread is active, this executes detection on frames continuously.
     * The next pending frame is either immediately available or hasn't been received yet. Once it
     * is available, we transfer the frame info to local variables and run detection on that frame.
     * It immediately loops back for the next frame without pausing.
     *
     * <p>If detection takes longer than the time in between new frames from the camera, this will
     * mean that this loop will run without ever waiting on a frame, avoiding any context switching
     * or frame acquisition time latency.
     *
     * <p>If you find that this is using more CPU than you'd like, you should probably decrease the
     * FPS setting above to allow for some idle time in between frames.
     */
    @SuppressLint("InlinedApi")
    @SuppressWarnings({"GuardedBy", "ByteBufferBackingArray"})
    @Override
    public void run() {
      ByteBuffer data;

      while (true) {
        synchronized (lock) {
          while (active && (pendingFrameData == null)) {
            try {
              // Wait for the next frame to be received from the camera, since we
              // don't have it yet.
              lock.wait();
            } catch (InterruptedException e) {
              Log.d(TAG, "Frame processing loop terminated.", e);
              return;
            }
          }

          if (!active) {
            // Exit the loop once this camera source is stopped or released.  We check
            // this here, immediately after the wait() above, to handle the case where
            // setActive(false) had been called, triggering the termination of this
            // loop.
            return;
          }

          // Hold onto the frame data locally, so that we can use this for detection
          // below.  We need to clear pendingFrameData to ensure that this buffer isn't
          // recycled back to the camera before we are done using that data.
          data = pendingFrameData;
          pendingFrameData = null;
        }

        // The code below needs to run outside of synchronization, because this will allow
        // the camera to add pending frame(s) while we are running detection on the current
        // frame.

        try {
          synchronized (processorLock) {
            frameProcessor.processByteBuffer(
                data,
                new FrameMetadata.Builder()
                    .setWidth(previewSize.getWidth())
                    .setHeight(previewSize.getHeight())
                    .setRotation(rotationDegrees)
                    .build(),
                graphicOverlay);
          }
        } catch (Exception t) {
          Log.e(TAG, "Exception thrown from receiver.", t);
        } finally {
          camera.addCallbackBuffer(data.array());
        }
      }
    }
  }

  /** Cleans up graphicOverlay and child classes can do their cleanups as well . */
  private void cleanScreen() {
    graphicOverlay.clear();
  }
}

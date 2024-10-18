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

package com.google.mlkit.vision.demo.kotlin.posedetector

import android.content.Context
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.odml.image.MlImage
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifierProcessor
import com.google.mlkit.vision.demo.kotlin.VisionProcessorBase
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase
import java.util.ArrayList
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/** A processor to run pose detector. */
// 포즈 감지기 프로세서를 정의하는 클래스
class PoseDetectorProcessor(
  private val context: Context, // 앱의 Context
  options: PoseDetectorOptionsBase, // 포즈 감지기 옵션
  private val showInFrameLikelihood: Boolean, // 포즈 감지 시 프레임 내의 신뢰도를 표시할지 여부
  private val visualizeZ: Boolean, // 포즈의 Z축(깊이)을 시각화할지 여부
  private val rescaleZForVisualization: Boolean, // Z값을 시각화할 때 재조정할지 여부
  private val runClassification: Boolean, // 포즈 분류를 실행할지 여부
  private val isStreamMode: Boolean // 스트리밍 모드 여부 (라이브 피드에 사용되는지 여부)
) : VisionProcessorBase<PoseDetectorProcessor.PoseWithClassification>(context) {

  // 포즈 감지 객체
  private val detector: PoseDetector
  // 포즈 분류를 수행할 Executor (단일 스레드)
  private val classificationExecutor: Executor

  // 포즈 분류 프로세서 (분류가 필요한 경우 생성)
  private var poseClassifierProcessor: PoseClassifierProcessor? = null

  /** 포즈와 분류 결과를 보관하는 내부 클래스 */
  class PoseWithClassification(val pose: Pose, val classificationResult: List<String>)

  init {
    // 포즈 감지기를 생성하여 초기화
    detector = PoseDetection.getClient(options)
    classificationExecutor = Executors.newSingleThreadExecutor()
  }

  // 리소스를 해제하기 위한 함수
  override fun stop() {
    super.stop()
    detector.close() // 포즈 감지기 종료
  }

  // InputImage 형식의 이미지를 포즈 감지하는 함수
  override fun detectInImage(image: InputImage): Task<PoseWithClassification> {
    return detector
      .process(image) // 이미지를 처리하여 포즈를 감지
      .continueWith(
        classificationExecutor, // 분류를 수행할 Executor 설정
        { task ->
          val pose = task.getResult() // 포즈 감지 결과 가져오기
          var classificationResult: List<String> = ArrayList()
          if (runClassification) { // 분류를 수행해야 하는 경우
            if (poseClassifierProcessor == null) {
              // 분류 프로세서가 없다면 생성
              poseClassifierProcessor = PoseClassifierProcessor(context, isStreamMode)
            }
            // 포즈 분류 결과 가져오기
            classificationResult = poseClassifierProcessor!!.getPoseResult(pose)
          }
          PoseWithClassification(pose, classificationResult) // 포즈와 분류 결과를 포함하는 객체 생성
        }
      )
  }

  // MlImage 형식의 이미지를 포즈 감지하는 함수 (InputImage와 동일한 방식으로 처리)
  override fun detectInImage(image: MlImage): Task<PoseWithClassification> {
    return detector
      .process(image)
      .continueWith(
        classificationExecutor,
        { task ->
          val pose = task.getResult()
          var classificationResult: List<String> = ArrayList()
          if (runClassification) {
            if (poseClassifierProcessor == null) {
              poseClassifierProcessor = PoseClassifierProcessor(context, isStreamMode)
            }
            classificationResult = poseClassifierProcessor!!.getPoseResult(pose)
          }
          PoseWithClassification(pose, classificationResult)
        }
      )
  }

  // 포즈 감지 및 분류 성공 시 호출되는 함수
  override fun onSuccess(
    poseWithClassification: PoseWithClassification, // 감지된 포즈 및 분류 결과
    graphicOverlay: GraphicOverlay // 그래픽 오버레이 객체
  ) {
    // 그래픽 오버레이에 포즈 결과를 추가하여 화면에 표시
    graphicOverlay.add(
      PoseGraphic(
        graphicOverlay,
        poseWithClassification.pose,
        showInFrameLikelihood,
        visualizeZ,
        rescaleZForVisualization,
        poseWithClassification.classificationResult
      )
    )
  }

  // 포즈 감지 실패 시 호출되는 함수
  override fun onFailure(e: Exception) {
    Log.e(TAG, "Pose detection failed!", e) // 오류 로그 출력
  }

  // MlImage 사용 여부를 반환하는 함수 (기본적으로 MlImage를 사용)
  override fun isMlImageEnabled(context: Context?): Boolean {
    // 기본적으로 MlImage를 사용, InputImage로 변경하려면 false 반환
    return true
  }

  // 클래스 내 상수 정의
  companion object {
    private val TAG = "PoseDetectorProcessor" // 로그 태그
  }
}


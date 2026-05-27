#!/usr/bin/env bash
# 동작: 환경 빌드하고 랜덤 움직임
# TwoArmThreading 환경을 생성하고 랜덤 액션으로 step 수행 (MuJoCo 뷰어에 렌더링)
# 목적: dexmimicgen/robosuite 환경이 정상 로드되는지 확인하는 sanity check
# --render 빼면 headless 실행

python scripts/demo_random_action.py --env TwoArmThreading --render

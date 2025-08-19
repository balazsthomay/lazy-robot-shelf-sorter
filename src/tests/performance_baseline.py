#!/usr/bin/env python3
"""
Performance Baseline - M4 Pro performance metrics
Part of Phase 1: Foundation - Milestone 1

"""

import time
from walking_skeleton import WalkingSkeleton
from robot_validation import validate_robot_integration


def establish_baseline() -> dict:
    """Function to measure what matters"""
    print("📊 Establishing M4 Pro performance baseline...")
    
    results = {}
    
    # Test 1: Walking skeleton performance
    start_time = time.time()
    skeleton = WalkingSkeleton()
    skeleton_success = skeleton.run_validation()
    results['walking_skeleton_time'] = time.time() - start_time
    results['walking_skeleton_success'] = skeleton_success
    
    # Test 2: Robot integration performance  
    start_time = time.time()
    robot_success = validate_robot_integration()
    results['robot_integration_time'] = time.time() - start_time
    results['robot_integration_success'] = robot_success
    
    # Overall results
    results['total_time'] = results['walking_skeleton_time'] + results['robot_integration_time']
    results['all_tests_passed'] = skeleton_success and robot_success
    
    return results


def print_baseline_report(results: dict):
    """Results display"""
    print("\n📊 M4 PRO PERFORMANCE BASELINE")
    print("=" * 40)
    print(f"Walking skeleton: {results['walking_skeleton_time']:.3f}s ({'✅' if results['walking_skeleton_success'] else '❌'})")
    print(f"Robot integration: {results['robot_integration_time']:.3f}s ({'✅' if results['robot_integration_success'] else '❌'})")
    print(f"Total time: {results['total_time']:.3f}s")
    print(f"All tests passed: {'✅' if results['all_tests_passed'] else '❌'}")
    
    # YAGNI: Simple success criteria
    if results['total_time'] < 5.0 and results['all_tests_passed']:
        print("\n🎉 MILESTONE 1 COMPLETE - Ready for Milestone 2!")
        return True
    else:
        print("\n⚠️  Performance issues - optimization needed")
        return False


def main():
    """Main function"""
    results = establish_baseline()
    success = print_baseline_report(results)
    return success


if __name__ == "__main__":
    main()
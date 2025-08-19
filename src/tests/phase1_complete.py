#!/usr/bin/env python3
"""
Phase 1 Complete - Foundation validation
Validates all components work together
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.performance_baseline import establish_baseline
from tests.milestone2_test import test_environment_stability  
from tests.milestone3_test import test_robot_shelf_integration


def validate_phase1_complete() -> bool:
    """Comprehensive Phase 1 validation"""
    print("🚀 PHASE 1: FOUNDATION - COMPLETE VALIDATION")
    print("=" * 50)
    
    results = {}
    start_time = time.time()
    
    # Test 1: Performance baseline
    print("\n1. Performance Baseline Test...")
    baseline_results = establish_baseline()
    results['baseline'] = baseline_results['all_tests_passed']
    
    # Test 2: Environment stability
    print("\n2. Environment Stability Test...")
    stability_result = test_environment_stability()
    results['stability'] = stability_result
    
    # Test 3: Robot integration
    print("\n3. Robot Integration Test...")
    robot_result = test_robot_shelf_integration()
    results['robot'] = robot_result
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 50)
    print("📊 PHASE 1 FOUNDATION - FINAL RESULTS")
    print("=" * 50)
    print(f"Performance baseline: {'✅ PASS' if results['baseline'] else '❌ FAIL'}")
    print(f"Environment stability: {'✅ PASS' if results['stability'] else '❌ FAIL'}")
    print(f"Robot integration: {'✅ PASS' if results['robot'] else '❌ FAIL'}")
    print(f"Total validation time: {total_time:.3f}s")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 PHASE 1: FOUNDATION COMPLETE!")
        print("✅ Ready for Phase 2: Vision System")
        print("\nFoundation Components Successfully Validated:")
        print("- PyBullet simulation environment")
        print("- Configurable shelf system (1-3 shelves)")
        print("- Robot controller with workspace analysis")
        print("- Physics stability with multiple objects")
        print("- Component interfaces and integration")
    else:
        print("\n❌ Phase 1 validation failed")
        print("Some components need attention before Phase 2")
    
    return all_passed


def main():
    """Run complete Phase 1 validation"""
    return validate_phase1_complete()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
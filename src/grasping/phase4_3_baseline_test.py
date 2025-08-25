"""
Phase 4.3 Baseline Testing Framework

Systematic testing of grasp success rates across 10 diverse YCB/GSO objects
to establish quantified baseline performance metrics.
"""

import pybullet as p
import numpy as np
import time
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

import sys
from pathlib import Path

try:
    # Try relative imports (when run as module)
    from .predictor import GGCNNPredictor
    from .coordinate_transforms import VisionSystemIntegrator
    from .planner import MotionPlanner
    from .executor import GraspExecutor, ExecutionResult, ExecutionMetrics
except ImportError:
    # Add project root to path when running directly
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.grasping.predictor import GGCNNPredictor
    from src.grasping.coordinate_transforms import VisionSystemIntegrator
    from src.grasping.planner import MotionPlanner
    from src.grasping.executor import GraspExecutor, ExecutionResult, ExecutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class TestObject:
    """Configuration for test object."""
    name: str
    path: str
    category: str           # Box, Can, Fruit, Tool, etc.
    difficulty: str         # Easy, Medium, Hard
    expected_success_rate: float  # Expected baseline success rate
    position: List[float]   # Default position on table
    orientation: List[float] # Default orientation (quaternion)
    scale: float = 1.0      # Scaling factor if needed

@dataclass
class TestResult:
    """Result from testing a single object."""
    object_name: str
    object_category: str
    attempts: int
    successes: int
    success_rate: float
    avg_force: float
    avg_lift_height: float
    avg_execution_time: float
    avg_stability: float
    avg_contacts: float
    metrics: List[ExecutionMetrics]
    error_messages: List[str]

@dataclass
class BaselineReport:
    """Complete baseline testing report."""
    overall_success_rate: float
    total_attempts: int
    total_successes: int
    object_results: List[TestResult]
    category_breakdown: Dict[str, float]
    difficulty_breakdown: Dict[str, float]
    execution_timestamp: str
    model_performance: Dict[str, float]


class BaselineTester:
    """
    Automated baseline testing framework for Phase 4.3.
    
    Tests 10 diverse objects to establish quantified success rates
    and identify areas for improvement.
    """
    
    def __init__(self):
        """Initialize baseline tester."""
        self.test_objects = self._define_test_objects()
        self.results_dir = Path("results/baseline_tests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.attempts_per_object = 5  # Number of attempts per object
        self.headless = False  # Set to True for batch testing
        
        logger.info(f"Initialized baseline tester with {len(self.test_objects)} test objects")
    
    def _define_test_objects(self) -> List[TestObject]:
        """
        Define diverse set of 10 test objects representing different grasp challenges.
        
        Returns:
            List of test object configurations
        """
        return [
            # Easy objects (cylindrical/simple shapes)
            TestObject(
                name="master_chef_can",
                path="assets/ycb/002_master_chef_can/tsdf/textured.obj",
                category="Can",
                difficulty="Easy",
                expected_success_rate=0.8,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="tomato_soup_can", 
                path="assets/ycb/005_tomato_soup_can/tsdf/textured.obj",
                category="Can",
                difficulty="Easy",
                expected_success_rate=0.75,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            
            # Medium objects (boxes, bottles)
            TestObject(
                name="cracker_box",
                path="assets/ycb/003_cracker_box/tsdf/textured.obj", 
                category="Box",
                difficulty="Medium",
                expected_success_rate=0.6,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="sugar_box",
                path="assets/ycb/004_sugar_box/tsdf/textured.obj",
                category="Box", 
                difficulty="Medium",
                expected_success_rate=0.55,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="mustard_bottle",
                path="assets/ycb/006_mustard_bottle/tsdf/textured.obj",
                category="Bottle",
                difficulty="Medium", 
                expected_success_rate=0.65,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="bleach_cleanser",
                path="assets/ycb/021_bleach_cleanser/tsdf/textured.obj",
                category="Bottle",
                difficulty="Medium",
                expected_success_rate=0.6,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            
            # Hard objects (irregular shapes, small, fragile)
            TestObject(
                name="banana",
                path="assets/ycb/011_banana/tsdf/textured.obj",
                category="Fruit", 
                difficulty="Hard",
                expected_success_rate=0.3,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="mug",
                path="assets/ycb/025_mug/tsdf/textured.obj",
                category="Dishware",
                difficulty="Hard",
                expected_success_rate=0.4,
                position=[0.0, 0.2, 0.68], 
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="fork",
                path="assets/ycb/030_fork/tsdf/textured.obj",
                category="Tool",
                difficulty="Hard",
                expected_success_rate=0.25,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            ),
            TestObject(
                name="spatula",
                path="assets/ycb/033_spatula/tsdf/textured.obj",
                category="Tool",
                difficulty="Hard", 
                expected_success_rate=0.2,
                position=[0.0, 0.2, 0.68],
                orientation=[0, 0, 0, 1]
            )
        ]
    
    def setup_simulation(self):
        """Setup PyBullet simulation for testing."""
        # Use headless mode for batch testing, GUI for observation
        physics_client = p.connect(p.DIRECT if self.headless else p.GUI)
        
        if not self.headless:
            p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=45, 
                                       cameraPitch=-30, cameraTargetPosition=[0.0, 0.2, 0.6])
        
        # Add PyBullet data path
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        plane_id = p.loadURDF("plane.urdf")
        
        # Load Franka Panda robot
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.5, 0.2, 0.0],
            useFixedBase=True
        )
        
        # Set robot to starting position
        joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
        for i, pos in enumerate(joint_positions):
            p.resetJointState(robot_id, i, pos)
        
        # Create table surface
        table_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.03])
        table_body = p.createMultiBody(0, table_id, -1, [0.0, 0.2, 0.60], [0, 0, 0, 1])
        p.changeVisualShape(table_body, -1, rgbaColor=[0.8, 0.8, 0.6, 1.0])
        
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        return physics_client, robot_id
    
    def setup_camera(self):
        """Setup camera for RGB-D capture."""
        camera_pos = [0.2, -0.3, 1.2]
        camera_target = [0.0, 0.2, 0.68]
        camera_up = [0, 0, 1]
        
        return {
            'position': camera_pos,
            'target': camera_target,
            'up': camera_up,
            'width': 640,
            'height': 480,
            'fov': 60,
            'aspect': 640 / 480,
            'near': 0.1,
            'far': 10.0
        }
    
    def load_object(self, test_obj: TestObject) -> Optional[int]:
        """
        Load test object into simulation.
        
        Args:
            test_obj: Test object configuration
            
        Returns:
            Object body ID or None if loading failed
        """
        try:
            # Check if object file exists
            obj_path = Path(test_obj.path)
            if not obj_path.exists():
                logger.error(f"Object file not found: {obj_path}")
                return None
            
            # Load object using URDF conversion if needed
            # For now, create simple geometric approximation
            if "can" in test_obj.name.lower():
                # Cylindrical objects
                obj_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.032, height=0.10)
                body_id = p.createMultiBody(0.3, obj_id, -1, test_obj.position, test_obj.orientation)
                p.changeVisualShape(body_id, -1, rgbaColor=[0.8, 0.2, 0.2, 1.0])
                
            elif "box" in test_obj.name.lower():
                # Box objects
                obj_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.06, 0.08])
                body_id = p.createMultiBody(0.4, obj_id, -1, test_obj.position, test_obj.orientation)
                p.changeVisualShape(body_id, -1, rgbaColor=[0.2, 0.8, 0.2, 1.0])
                
            elif "bottle" in test_obj.name.lower():
                # Bottle objects (cylinder with different proportions)
                obj_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.028, height=0.15)
                body_id = p.createMultiBody(0.35, obj_id, -1, test_obj.position, test_obj.orientation)
                p.changeVisualShape(body_id, -1, rgbaColor=[0.2, 0.2, 0.8, 1.0])
                
            elif "mug" in test_obj.name.lower():
                # Mug (cylinder with handle - simplified)
                obj_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.035, height=0.08)
                body_id = p.createMultiBody(0.25, obj_id, -1, test_obj.position, test_obj.orientation)
                p.changeVisualShape(body_id, -1, rgbaColor=[0.8, 0.8, 0.2, 1.0])
                
            else:
                # Generic objects (small ellipsoid/sphere)
                obj_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
                body_id = p.createMultiBody(0.15, obj_id, -1, test_obj.position, test_obj.orientation)
                p.changeVisualShape(body_id, -1, rgbaColor=[0.8, 0.4, 0.8, 1.0])
            
            logger.debug(f"Loaded test object: {test_obj.name} with ID {body_id}")
            return body_id
            
        except Exception as e:
            logger.error(f"Failed to load object {test_obj.name}: {e}")
            return None
    
    def capture_rgbd(self, camera_params):
        """Capture RGB-D images from camera."""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_params['position'],
            cameraTargetPosition=camera_params['target'],
            cameraUpVector=camera_params['up']
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=camera_params['fov'],
            aspect=camera_params['aspect'],
            nearVal=camera_params['near'],
            farVal=camera_params['far']
        )
        
        width, height = camera_params['width'], camera_params['height']
        
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        depth_array = np.array(depth_img).reshape(height, width)
        
        # Convert depth to meters
        near, far = camera_params['near'], camera_params['far']
        depth_meters = far * near / (far - (far - near) * depth_array)
        
        return rgb_array, depth_meters, view_matrix, proj_matrix
    
    def test_single_object(self, test_obj: TestObject, 
                          predictor: GGCNNPredictor,
                          grasp_executor: GraspExecutor,
                          camera_params: dict) -> TestResult:
        """
        Test grasping performance on a single object.
        
        Args:
            test_obj: Test object configuration
            predictor: GG-CNN predictor
            grasp_executor: Grasp executor
            camera_params: Camera parameters
            
        Returns:
            Test result with detailed metrics
        """
        print(f"\\n=== Testing {test_obj.name.upper()} ({test_obj.category}, {test_obj.difficulty}) ===")
        
        successes = 0
        all_metrics = []
        error_messages = []
        
        for attempt in range(self.attempts_per_object):
            print(f"\\n--- Attempt {attempt + 1}/{self.attempts_per_object} ---")
            
            try:
                # Load object for this attempt
                obj_id = self.load_object(test_obj)
                if obj_id is None:
                    error_messages.append(f"Failed to load object for attempt {attempt + 1}")
                    continue
                
                # Let simulation settle
                for _ in range(100):
                    p.stepSimulation()
                
                # Capture RGB-D
                rgb_img, depth_img, _, _ = self.capture_rgbd(camera_params)
                
                # Get grasp predictions
                grasp_poses = predictor.predict(rgb_img, depth_img)
                if not grasp_poses:
                    error_messages.append(f"No grasp predictions for attempt {attempt + 1}")
                    p.removeBody(obj_id)
                    continue
                
                print(f"   Generated {len(grasp_poses)} grasp candidates")
                
                # Execute grasp with retry
                result = grasp_executor.execute_with_retry(grasp_poses)
                
                # Record metrics
                if result.metrics:
                    all_metrics.append(result.metrics)
                
                # Check success
                if result.success:
                    successes += 1
                    print(f"   🎯 SUCCESS: Grasp completed successfully!")
                else:
                    print(f"   ❌ FAILED: {result.error_message or 'Unknown failure'}")
                    if result.error_message:
                        error_messages.append(result.error_message)
                
                # Clean up object
                p.removeBody(obj_id)
                
            except Exception as e:
                error_message = f"Exception in attempt {attempt + 1}: {str(e)}"
                error_messages.append(error_message)
                logger.error(error_message)
        
        # Calculate averages
        success_rate = successes / self.attempts_per_object
        avg_force = np.mean([m.max_gripper_force for m in all_metrics]) if all_metrics else 0.0
        avg_lift = np.mean([m.lift_height_achieved for m in all_metrics]) if all_metrics else 0.0
        avg_time = np.mean([m.execution_time for m in all_metrics]) if all_metrics else 0.0
        avg_stability = np.mean([m.grasp_stability_score for m in all_metrics]) if all_metrics else 0.0
        avg_contacts = np.mean([m.contact_points for m in all_metrics]) if all_metrics else 0.0
        
        result = TestResult(
            object_name=test_obj.name,
            object_category=test_obj.category,
            attempts=self.attempts_per_object,
            successes=successes,
            success_rate=success_rate,
            avg_force=avg_force,
            avg_lift_height=avg_lift,
            avg_execution_time=avg_time,
            avg_stability=avg_stability,
            avg_contacts=avg_contacts,
            metrics=all_metrics,
            error_messages=error_messages
        )
        
        print(f"\\n📊 {test_obj.name.upper()} RESULTS:")
        print(f"   Success Rate: {success_rate:.1%} ({successes}/{self.attempts_per_object})")
        print(f"   Avg Force: {avg_force:.1f}N | Avg Lift: {avg_lift:.3f}m")
        print(f"   Avg Stability: {avg_stability:.2f} | Avg Contacts: {avg_contacts:.1f}")
        
        return result
    
    def run_baseline_test(self) -> BaselineReport:
        """
        Run complete baseline testing across all objects.
        
        Returns:
            Comprehensive baseline report
        """
        print("🚀 PHASE 4.3 BASELINE TESTING FRAMEWORK")
        print("=" * 50)
        
        # Setup simulation
        physics_client, robot_id = self.setup_simulation()
        camera_params = self.setup_camera()
        
        try:
            # Initialize components
            predictor = GGCNNPredictor(device='cpu', max_predictions=3)
            
            # Load model
            model_path = Path(__file__).parent.parent.parent / "data" / "models" / "ggcnn_epoch_23_cornell_statedict.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            predictor.load_model(str(model_path))
            
            # Setup coordinate transformer
            vision_integrator = VisionSystemIntegrator()
            transformer = vision_integrator.setup_from_camera_system(
                camera_pos=np.array(camera_params['position']),
                camera_target=np.array(camera_params['target']),
                camera_up=np.array(camera_params['up']),
                fov=camera_params['fov'],
                aspect=camera_params['aspect'],
                near=camera_params['near'],
                far=camera_params['far'],
                width=camera_params['width'],
                height=camera_params['height']
            )
            predictor.set_coordinate_transformer(transformer)
            
            # Initialize motion planner and executor
            motion_planner = MotionPlanner(robot_id)
            grasp_executor = GraspExecutor(robot_id, motion_planner)
            
            print(f"✅ All components initialized successfully")
            print(f"📋 Testing {len(self.test_objects)} objects with {self.attempts_per_object} attempts each")
            
            # Run tests for each object
            object_results = []
            
            for i, test_obj in enumerate(self.test_objects):
                print(f"\\n[{i+1}/{len(self.test_objects)}] Starting test for {test_obj.name}")
                
                result = self.test_single_object(
                    test_obj, predictor, grasp_executor, camera_params
                )
                object_results.append(result)
                
                if not self.headless:
                    time.sleep(2)  # Brief pause for observation
            
            # Calculate overall statistics
            total_attempts = sum(r.attempts for r in object_results)
            total_successes = sum(r.successes for r in object_results)
            overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0
            
            # Category breakdown
            category_stats = {}
            for category in set(obj.category for obj in self.test_objects):
                cat_results = [r for r in object_results if r.object_category == category]
                cat_attempts = sum(r.attempts for r in cat_results)
                cat_successes = sum(r.successes for r in cat_results)
                category_stats[category] = cat_successes / cat_attempts if cat_attempts > 0 else 0.0
            
            # Difficulty breakdown
            difficulty_stats = {}
            for difficulty in set(obj.difficulty for obj in self.test_objects):
                diff_objects = [obj for obj in self.test_objects if obj.difficulty == difficulty]
                diff_results = [r for r in object_results if r.object_name in [obj.name for obj in diff_objects]]
                diff_attempts = sum(r.attempts for r in diff_results)
                diff_successes = sum(r.successes for r in diff_results)
                difficulty_stats[difficulty] = diff_successes / diff_attempts if diff_attempts > 0 else 0.0
            
            # Model performance metrics
            all_metrics = [m for r in object_results for m in r.metrics]
            model_performance = {
                'avg_execution_time': np.mean([m.execution_time for m in all_metrics]) if all_metrics else 0.0,
                'avg_force_consistency': np.mean([m.force_consistency for m in all_metrics]) if all_metrics else 0.0,
                'avg_stability_score': np.mean([m.grasp_stability_score for m in all_metrics]) if all_metrics else 0.0
            }
            
            # Create final report
            report = BaselineReport(
                overall_success_rate=overall_success_rate,
                total_attempts=total_attempts,
                total_successes=total_successes,
                object_results=object_results,
                category_breakdown=category_stats,
                difficulty_breakdown=difficulty_stats,
                execution_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_performance=model_performance
            )
            
            return report
            
        finally:
            # Cleanup
            p.disconnect(physics_client)
    
    def print_report(self, report: BaselineReport):
        """Print comprehensive baseline report."""
        print("\\n" + "=" * 60)
        print("📈 PHASE 4.3 BASELINE TESTING REPORT")
        print("=" * 60)
        
        print(f"\\n🎯 OVERALL PERFORMANCE:")
        print(f"   Success Rate: {report.overall_success_rate:.1%} ({report.total_successes}/{report.total_attempts})")
        
        # Determine if we meet Phase 4.3 success gate (>50%)
        success_gate_met = report.overall_success_rate >= 0.5
        gate_status = "✅ PASSED" if success_gate_met else "❌ FAILED" 
        print(f"   Phase 4.3 Success Gate (>50%): {gate_status}")
        
        print(f"\\n📊 CATEGORY BREAKDOWN:")
        for category, rate in report.category_breakdown.items():
            print(f"   {category}: {rate:.1%}")
        
        print(f"\\n🎚️ DIFFICULTY BREAKDOWN:")
        for difficulty, rate in report.difficulty_breakdown.items():
            print(f"   {difficulty}: {rate:.1%}")
        
        print(f"\\n⚡ MODEL PERFORMANCE:")
        print(f"   Avg Execution Time: {report.model_performance['avg_execution_time']:.1f}s")
        print(f"   Avg Force Consistency: {report.model_performance['avg_force_consistency']:.2f}")
        print(f"   Avg Stability Score: {report.model_performance['avg_stability_score']:.2f}")
        
        print(f"\\n🔍 DETAILED OBJECT RESULTS:")
        for result in sorted(report.object_results, key=lambda x: x.success_rate, reverse=True):
            status_emoji = "🟢" if result.success_rate >= 0.7 else "🟡" if result.success_rate >= 0.4 else "🔴"
            print(f"   {status_emoji} {result.object_name:<15} ({result.object_category:<8}): "
                  f"{result.success_rate:.1%} | Force: {result.avg_force:.1f}N | "
                  f"Lift: {result.avg_lift_height:.3f}m")
        
        # Recommendations
        print(f"\\n💡 RECOMMENDATIONS:")
        if not success_gate_met:
            print("   🔥 URGENT: Baseline <50% - Data Generation Pipeline Required")
            print("   📈 Generate 1000+ training scenarios for GG-CNN fine-tuning")
            print("   🎯 Focus on low-performing categories and difficult objects")
        else:
            print("   ✅ Baseline meets requirements - continue to Phase 4.4")
            print("   🔧 Consider optimizations for consistently failing objects")
        
        print(f"\\n📅 Test completed: {report.execution_timestamp}")
        print("=" * 60)
    
    def save_report(self, report: BaselineReport, filename: Optional[str] = None):
        """Save baseline report to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_report_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert dataclasses to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\\n💾 Report saved to: {filepath}")


def main():
    """Run Phase 4.3 baseline testing."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create and run baseline tester
    tester = BaselineTester()
    
    # Run the test
    report = tester.run_baseline_test()
    
    # Print and save results
    tester.print_report(report)
    tester.save_report(report)
    
    # Return success gate status
    success_gate_met = report.overall_success_rate >= 0.5
    if success_gate_met:
        print("\\n🎉 Phase 4.3 Success Gate PASSED - Ready for Phase 4.4!")
    else:
        print("\\n⚠️ Phase 4.3 Success Gate FAILED - Data Generation Required!")
    
    return report


if __name__ == "__main__":
    main()
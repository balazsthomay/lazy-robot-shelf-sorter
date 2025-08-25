#!/usr/bin/env python3

import pybullet as p
import numpy as np
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_simple_grasp():
    """Test robot grasp execution with direct joint control - no vision needed."""
    
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0.2, 0.68])
    
    # Load environment
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.7, 0.2, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, 3.14159]), useFixedBase=True)
    
    # Create table with both collision and visual shapes
    table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.03])
    table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.03], rgbaColor=[0.8, 0.8, 0.6, 1.0])
    table_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_collision, 
                                   baseVisualShapeIndex=table_visual, basePosition=[0.0, 0.2, 0.60])
    
    # Create object with both collision and visual shapes  
    obj_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.032, height=0.10)
    obj_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.032, length=0.10, rgbaColor=[0.8, 0.2, 0.2, 1.0])
    # Place object where robot can actually reach - closer to robot's workspace
    # Use X=0.25 (where robot can reach), Y=0.2 (table center), Z=0.73 to drop onto table
    obj_body = p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=obj_collision,
                                baseVisualShapeIndex=obj_visual, basePosition=[0.25, 0.200, 0.73])
    
    p.setGravity(0, 0, -9.81)
    
    # Let physics settle and check object position
    for i in range(100):
        p.stepSimulation()
        if i % 20 == 0:  # Check every 20 steps
            obj_pos = p.getBasePositionAndOrientation(obj_body)[0]
            print(f"Step {i}: Object at z={obj_pos[2]:.3f}")
    
    # Final check
    obj_pos_settled = p.getBasePositionAndOrientation(obj_body)[0]
    print(f"Object settled at: [{obj_pos_settled[0]:.3f}, {obj_pos_settled[1]:.3f}, {obj_pos_settled[2]:.3f}]")
    
    print("🤖 SIMPLE GRASP TEST")
    print("=" * 40)
    
    # Set robot to home position smoothly  
    print("1. Moving to home position...")
    joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.08, 0.08]  # Open gripper wide
    
    # Move to home position with smooth control
    for step in range(120):
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500,
                maxVelocity=2.3  # Faster speed for home position
            )
        # Set gripper position using correct finger joints
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, joint_positions[7], force=200)  # finger 1 
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, joint_positions[8], force=200) # finger 2
        p.stepSimulation()
        time.sleep(0.01)
    
    # Use IK to reach above the object - approach from the side to avoid table collision
    print("2. Using IK to reach pre-grasp position...")
    
    # First, ensure gripper is wide open
    print("   Opening gripper wide...")
    for step in range(30):
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.08, force=200)  # finger 1
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.08, force=200) # finger 2
        p.stepSimulation()
        time.sleep(0.01)
    
    object_pos = [0.25, 0.200, 0.68]  # Updated object position (on table)
    # Approach from robot's side (robot is at [0.5, 0.2, 0])
    pre_grasp_pos = [0.25, 0.200, 0.75]  # Right above object, higher up
    pre_grasp_orn = [0, 0, 0, 1]
    
    pre_grasp_joints = p.calculateInverseKinematics(
        robot_id, 11, pre_grasp_pos, pre_grasp_orn,
        lowerLimits=[-2.8973, -2.5, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        upperLimits=[2.8973, 2.5, 2.8973, 3.0718, 2.8973, 6.7020, 2.8973],
        jointRanges=[5.7946, 5.0, 5.7946, 6.1436, 5.7946, 6.7195, 5.7946],
        maxNumIterations=100
    )
    
    print(f"Target pre-grasp position: {pre_grasp_pos}")
    
    # Use smooth joint control instead of instant reset
    for step in range(100):
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, 
                controlMode=p.POSITION_CONTROL,
                targetPosition=pre_grasp_joints[i],
                force=500,
                maxVelocity=1.0  # Medium speed movement
            )
        # Keep gripper open using correct finger joints
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.08, force=200)  # finger 1
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.08, force=200) # finger 2
        p.stepSimulation()
        time.sleep(0.01)
    
    # Verify end effector position
    ee_state = p.getLinkState(robot_id, 11)
    ee_pos = ee_state[0]
    print(f"End effector at: {ee_pos}")
    
    # Use IK to reach grasp position - directly at object
    print("3. Moving to grasp position...")
    grasp_pos = [0.25, 0.200, 0.68]  # Directly at object position
    grasp_orn = [0, 0, 0, 1]
    
    grasp_joints = p.calculateInverseKinematics(
        robot_id, 11, grasp_pos, grasp_orn,
        lowerLimits=[-2.8973, -2.5, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        upperLimits=[2.8973, 2.5, 2.8973, 3.0718, 2.8973, 6.7020, 2.8973],
        jointRanges=[5.7946, 5.0, 5.7946, 6.1436, 5.7946, 6.7195, 5.7946],
        maxNumIterations=100
    )
    
    # Move to grasp with collision detection
    print("   Moving down slowly until contact...")
    contact_detected = False
    for step in range(100):
        # Check for contact between gripper fingers and object
        contacts_finger1 = p.getContactPoints(bodyA=robot_id, bodyB=obj_body, linkIndexA=9)
        contacts_finger2 = p.getContactPoints(bodyA=robot_id, bodyB=obj_body, linkIndexA=10)
        if contacts_finger1 or contacts_finger2:
            print(f"   ✅ Contact detected at step {step}!")
            contact_detected = True
            break
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=grasp_joints[i],
                force=500,
                maxVelocity=0.3)  # Slower speed for contact detection
# Keep gripper open using correct finger joints
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.08, force=200)  # finger 
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.08, force=200) # finger 
    
        p.stepSimulation()
        time.sleep(0.02)  # Slower for better contact detection
   
    if not contact_detected:
        print("   ⚠️  No contact detected, proceeding anyway...")
    
    # Verify we're at the object
    ee_state = p.getLinkState(robot_id, 11)
    ee_pos = ee_state[0]
    print(f"End effector at grasp: {ee_pos}")
    print(f"Object at: {object_pos}")
    
    # Close gripper slowly
    print("4. Closing gripper...")
    for step in range(50):
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.01, force=200)  # finger 1
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.01, force=200) # finger 2
        p.stepSimulation()
        time.sleep(0.02)
    
    # Check if object moved
    obj_pos_after = p.getBasePositionAndOrientation(obj_body)[0]
    print(f"Object position after grasp: {obj_pos_after}")
    
    # Lift object using IK
    print("5. Lifting object...")
    lift_pos = [object_pos[0], object_pos[1], object_pos[2] + 0.20]  # Lift 20cm
    lift_orn = [0, 0, 0, 1]
    
    lift_joints = p.calculateInverseKinematics(
        robot_id, 11, lift_pos, lift_orn
    )
    
    # Lift slowly
    for step in range(150):
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=lift_joints[i],
                force=500,
                maxVelocity=0.6  # Faster lift
            )
        # Keep gripper closed using correct finger joints
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.01, force=200)  # finger 1
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.01, force=200) # finger 2
        p.stepSimulation()
        time.sleep(0.01)
    
    # Final object position
    obj_pos_final = p.getBasePositionAndOrientation(obj_body)[0]
    print(f"Object position after lift: {obj_pos_final}")
    
    # Check if object was lifted
    if obj_pos_final[2] > 0.75:
        print("✅ SUCCESS: Object was lifted!")
        return True
    else:
        print("❌ FAILED: Object was not lifted")
        return False

if __name__ == "__main__":
    success = test_simple_grasp()
    if success:
        print("\\n🎉 Robot can grasp and lift objects!")
    else:
        print("\\n❌ Robot failed to grasp object")
#!/usr/bin/env python3
"""
Test if Franka Panda URDF is available in PyBullet
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import pybullet_data

def test_franka_availability():
    """Test if Franka Panda URDF exists and can be loaded"""
    print("🔍 Testing Franka Panda URDF availability...")
    
    # Connect headless
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Test 1: Check if file exists
    data_path = pybullet_data.getDataPath()
    franka_path = os.path.join(data_path, "franka_panda", "panda.urdf")
    
    print(f"PyBullet data path: {data_path}")
    print(f"Looking for Franka at: {franka_path}")
    
    if os.path.exists(franka_path):
        print("✅ Franka Panda URDF file found!")
    else:
        print("❌ Franka Panda URDF file not found")
        
        # List what's available in the data directory
        print("\nAvailable robots in PyBullet data:")
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Look for URDF files in subdirectories
                try:
                    for subitem in os.listdir(item_path):
                        if subitem.endswith('.urdf'):
                            print(f"    📄 {subitem}")
                except:
                    pass
        
        p.disconnect()
        return False
    
    # Test 2: Try to load the URDF
    print("\n🤖 Testing Franka Panda URDF loading...")
    
    try:
        # Load ground first
        p.setGravity(0, 0, -9.81)
        ground_id = p.loadURDF("plane.urdf")
        
        # Try to load Franka Panda
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Get robot info
        num_joints = p.getNumJoints(robot_id)
        
        print(f"✅ Franka Panda loaded successfully!")
        print(f"   Robot ID: {robot_id}")
        print(f"   Number of joints: {num_joints}")
        
        # Get joint info
        print("\n📊 Joint information:")
        controllable_joints = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            type_names = {
                p.JOINT_REVOLUTE: "REVOLUTE",
                p.JOINT_PRISMATIC: "PRISMATIC", 
                p.JOINT_FIXED: "FIXED"
            }
            
            type_name = type_names.get(joint_type, f"UNKNOWN({joint_type})")
            print(f"   Joint {i}: {joint_name} ({type_name})")
            
            if joint_type != p.JOINT_FIXED:
                controllable_joints.append(i)
        
        print(f"\n✅ Controllable joints: {len(controllable_joints)}")
        print(f"   Joint indices: {controllable_joints}")
        
        # Test basic movement
        print("\n🔧 Testing basic joint control...")
        if controllable_joints:
            # Set a simple joint position
            p.setJointMotorControl2(robot_id, controllable_joints[0], 
                                  p.POSITION_CONTROL, targetPosition=0.5)
            
            # Step simulation
            for _ in range(10):
                p.stepSimulation()
            
            # Check joint state
            joint_state = p.getJointState(robot_id, controllable_joints[0])
            print(f"   Joint {controllable_joints[0]} position: {joint_state[0]:.3f}")
            print("✅ Basic joint control working!")
        
        p.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Franka Panda: {e}")
        p.disconnect()
        return False

if __name__ == "__main__":
    success = test_franka_availability()
    
    if success:
        print("\n🎉 Franka Panda is ready to use!")
        print("✅ Built-in PyBullet URDF works perfectly")
    else:
        print("\n⚠️ Franka Panda not available in built-in PyBullet")
        print("Will need to download from external source")
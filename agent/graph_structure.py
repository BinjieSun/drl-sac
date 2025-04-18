'''
Humanoid-v5
               torso
              /   |  \
             /    |   \
            /     |    \
   left_upper_arm |  right_upper_arm
          |       |        |
   left_lower_arm |  right_lower_arm
                  |
               lwaist
                  |
               pelvis
              /     \
      left_thigh     right_thigh
          |             |
      left_shin       right_shin
          |             |
      left_foot       right_foot
      

348 features
| Joint Name      | Joint Index | *qpos & *qvel Feature Index          | qfrc_actuator Index |
|-----------------|-------------|--------------------------------------|---------------------|
| torso           | 0           | *range(0, 5), *range(22, 28)         |                     |
| lwaist          | 1           | *range(5, 7), *range(28, 30)         |                     |
| pelvis          | 2           | 7, 30                                | 0, 1, 2             |
| right_thigh     | 3           | *range(8, 12), *range(31, 35)        | 3, 4, 5             |
| right_shin      | 4           | 11, 34                               | 6                   |
| right_foot      | 5           |                                      |                     |
| left_thigh      | 6           | *range(12, 16), *range(35, 39)       | 7, 8, 9             |
| left_shin       | 7           | 15, 38                               | 10                  |
| left_foot       | 8           |                                      |                     |
| right_upper_arm | 9           | *range(16, 19), *range(39, 42)       | 11, 12              |
| right_lower_arm | 10          | 18, 41                               | 13                  |
| left_upper_arm  | 11          | *range(19, 22), *range(42, 45)       | 14, 15              |
| left_lower_arm  | 12          | 21, 44                               | 16                  |

13 nodes

observation space as follows:

| Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
| --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
| 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
| 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 4   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_z                        | hinge | angle (rad)                |
| 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_y                        | hinge | angle (rad)                |
| 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)                |
| 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_x                      | hinge | angle (rad)                |
| 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_z                      | hinge | angle (rad)                |
| 10  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_y                      | hinge | angle (rad)                |
| 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)                |
| 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_x                       | hinge | angle (rad)                |
| 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_z                       | hinge | angle (rad)                |
| 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_y                       | hinge | angle (rad)                |
| 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)                |
| 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder1                  | hinge | angle (rad)                |
| 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder2                  | hinge | angle (rad)                |
| 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)                |
| 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder1                   | hinge | angle (rad)                |
| 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder2                   | hinge | angle (rad)                |
| 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)                |
| 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
| 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
| 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
| 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | angular velocity (rad/s)   |
| 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | angular velocity (rad/s)   |
| 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | angular velocity (rad/s)   |
| 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_x                      | hinge | angular velocity (rad/s)   |
| 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | angular velocity (rad/s)   |
| 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | angular velocity (rad/s)   |
| 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | angular velocity (rad/s)   |
| 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_x                       | hinge | angular velocity (rad/s)   |
| 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | angular velocity (rad/s)   |
| 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | angular velocity (rad/s)   |
| 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | angular velocity (rad/s)   |
| 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder1                  | hinge | angular velocity (rad/s)   |
| 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder2                  | hinge | angular velocity (rad/s)   |
| 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | angular velocity (rad/s)   |
| 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder1                   | hinge | angular velocity (rad/s)   |
| 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder2                   | hinge | angular velocity (rad/s)   |
| 44  | angular velocity of the angle between left upper arm and left_lower_arm                                         | -Inf | Inf | left_elbow                       | hinge | angular velocity (rad/s)   |

- *cinert (130 elements):* Mass and inertia of the rigid body parts relative to the center of mass,
(this is an intermediate result of the transition).
It has shape 13*10 (*nbody * 10*).
(cinert - inertia matrix and body mass offset and body mass)
- *cvel (78 elements):* Center of mass based velocity.
It has shape 13 * 6 (*nbody * 6*).
(com velocity - velocity x, y, z and angular velocity x, y, z)
- *qfrc_actuator (17 elements):* Constraint force generated as the actuator force at each joint.
This has shape `(17,)`  *(nv * 1)*.
- *cfrc_ext (78 elements):* This is the center of mass based external force on the body parts.
It has shape 13 * 6 (*nbody * 6*) and thus adds another 78 elements to the observation space.
(external forces - force x, y, z and torque x, y, z)

The action space is a `Box(-0.4, 0.4, (17,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    | --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
    | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
    | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
    | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
    | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
    | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
    | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
    | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
    | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |


'''

'''
HalfCheetah-v5
                           front_tip
                          /
                         /
                        /
      0 bthigh ---- 3 fthigh
      /               \
    /                  \
   /                    \
  1 bshin               4 fshin
   /                    \
  /                      \
 2 bfoot                 5 ffoot
 
6 nodes, 17 features
| Joint Name      | Joint Index | Feature Index          | Actuator Index |
|-----------------|-------------|------------------------|----------------|
| bthigh          | 0           | 0, 1, 8, 9, 10, 2, 11  | 17             |
| bshin           | 1           | 0, 1, 8, 9, 10, 3, 12  | 18             |
| bfoot           | 2           | 0, 1, 8, 9, 10, 4, 13  | 19             |
| fthigh          | 3           | 0, 1, 8, 9, 10, 5, 14  | 20             |
| fshin           | 4           | 0, 1, 8, 9, 10, 6, 15  | 21             |
| ffoot           | 5           | 0, 1, 8, 9, 10, 7, 16  | 22             |
'''

'''
Walker2d-v5

             torso
               |
               |
               |
      0 bthigh ---- 3 fthigh
      /               \
     /                 \
  1 bshin               4 fshin
   /                     \
  /                       \
 2 bfoot                  5 ffoot


6 nodes, 17 features
| Joint Name      | Joint Index | Feature Index          | Actuator Index |
|-----------------|-------------|------------------------|----------------|
| rthigh          | 0           | 0, 1, 8, 9, 10, 2, 11  | 17             |
| rshin           | 1           | 0, 1, 8, 9, 10, 3, 12  | 18             |
| rfoot           | 2           | 0, 1, 8, 9, 10, 4, 13  | 19             |
| lthigh          | 3           | 0, 1, 8, 9, 10, 5, 14  | 20             |
| lshin           | 4           | 0, 1, 8, 9, 10, 6, 15  | 21             |
| lfoot           | 5           | 0, 1, 8, 9, 10, 7, 16  | 22             |

'''
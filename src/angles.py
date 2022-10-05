import numpy as np

"""
CODE TO HOPEFULLY GENERATE EULER ANGLES OF MOUSE, GIVEN BASELINE AND CURRENT DLC POINTS

    expected input:
        baseline: in this case 4pts supplied as x,y coordinates (z=0 assumed)
        //NOTE// - baseline actually 3 length values!
        frame: 4pts at given time, supplied as x,y coordinates
        
    output:
        3 euler angles: roll (about x'), pitch (about y'), and yaw (about z')
        
    steps:
        1. compute the z coords of the frame points
        2. compute ON basis for the baseline
            - assumption: x', y' not strictly orthogonal but assume so and correct y'
        3. compute ON basis for the frame of interest
        4. find rotation matrix
        5. determine euler angles from this matrix!
        
    OK!          
"""

def get_z_coords(x_frame, x_baseline):

    """
    assume back midpt of head has z = 0

    :param x_frame: snout, le, re, tail - 4x2
    :param X_baseline: snout, le, re, tail - 4x2
    :return: x_frame_3d - 4x3
    """
    # 1. get norm of baseline sides
    s1_base = np.linalg.norm(x_baseline[0] - x_baseline[1])
    print(s1_base**2)
    s2_base = np.linalg.norm(x_baseline[1] - x_baseline[2])
    s3_base = np.linalg.norm(x_baseline[2] - x_baseline[0])

    # 2. get norm of frame sides
    s1_frame = np.linalg.norm(x_frame[0] - x_frame[1])
    print(s1_frame**2)
    s2_frame = np.linalg.norm(x_frame[1] - x_frame[2])
    s3_frame = np.linalg.norm(x_frame[2] - x_frame[0])

    # 3. compute z along each side of triange - pythagoras
    z_s1 = np.sqrt(s1_base**2 - s1_frame**2)
    cos = np.arccos((s1_frame / s1_base))
    print("z side 1: ", z_s1)
    z_s2 = np.sqrt(s2_base**2 - s2_frame**2)
    cos2 = np.arccos((s2_frame / s2_base))
    print("z side 2: ", z_s2)
    z_s3 = np.sqrt(s3_base**2 - s3_frame**2)
    cos3 = np.arccos((s3_frame / s3_base))
    print("z side 3: ", z_s3)

    # 4. get z values for each point back midpt has z=0
    #left_ear_z = - z_s2 / 2
    left_ear_z = 0
    print(left_ear_z)
    right_ear_z = z_s2
    print(right_ear_z)
    snout_z = left_ear_z + z_s1
    print(snout_z)
    snout_z_2 = right_ear_z + z_s3
    print(snout_z_2)

    # 5. check z is correct
    if np.abs(snout_z - snout_z_2) > 1e-4:
        print("get z not working huh", snout_z, snout_z_2)

        midpt_base = (x_baseline[1] + x_baseline[2]) / 2
        midpt_frame = (x_frame[1] + x_frame[2]) / 2

        h_axis_base = np.linalg.norm(x_baseline[0] - midpt_base)
        h_axis_frame = np.linalg.norm(x_frame[0] - midpt_frame)
        snout_z_3 = np.sqrt(h_axis_base**2 - h_axis_frame**2)
        print("snout z3: ", snout_z_3)
        print((snout_z_3-snout_z_2, snout_z_3-snout_z))

    else:
        print("Z good --------------------------")


    # 6. add z vals to x array
    snout_3d = np.append(x_frame[0], snout_z)
    left_ear_3d = np.append(x_frame[1], left_ear_z)
    right_ear_3d = np.append(x_frame[2], right_ear_z)
    tail_3d = np.append(x_frame[3], 0.0)

    x_frame_3d = np.concatenate((snout_3d, left_ear_3d, right_ear_3d, tail_3d), axis=0).reshape(4, 3)

    return x_frame_3d

"""
From Magnus Svensson Pierrau to Everyone:  04:18 PM
rotx = lambda angle : np.array([[1,0,0],[0, np.cos(angle), -np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
roty = lambda angle : np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
rotz = lambda angle : np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])

rot = lambda X , yaw , pitch , roll : ((X @ rotx(np.radians(roll)) @ roty(np.radians(pitch)))) @ rotz(np.radians(yaw))
"""

def get_ON_basis(x):
    midpt = (x[1] + x[2])/2

    u1 = (x[0] - midpt) / np.linalg.norm(x[0] - midpt)
    u2_star = (x[1] - x[2]) / np.linalg.norm(x[1] - x[2])

    #correct for non orthogonality
    u3_star = np.cross(u1, u2_star)
    u3 = u3_star / np.linalg.norm(u3_star)

    u2 = np.cross(u1, u3)

    #tests
    if (np.dot(u1, u2) > 1e-8 or np.dot(u2, u3) > 1e-8 or np.dot(u3, u1) > 1e-8):
        print("VECTORS NOT ORTHOGONAL!")
        print(np.dot(u1, u2), np.dot(u2, u3), np.dot(u3, u1))

    elif (np.dot(u1, u1) - 1.0 > 1e-8 or np.dot(u2, u2) - 1.0 > 1e-8 or np.dot(u3, u3) - 1.0 > 1e-8):
        print("vectors not normalized")
        print(np.dot(u1, u1), np.dot(u2, u2), np.dot(u3, u3))

    print(u1)
    basis = np.concatenate((u1, u2, u3), axis=0).reshape((3,3))
    print("Basis",basis)

    #want the basis as column vecs
    return basis.T

def get_linear_combination(basis, x):
    #scalars returns coefficient matrix of the new basis as linear comb of the old basis
    #this is either the rotation matrix we want, or its transpose, i think!

    scalars = np.zeros((3,3))
    scalars[0] = np.linalg.solve(basis, x[0])
    scalars[1] = np.linalg.solve(basis, x[1])
    scalars[2] = np.linalg.solve(basis, x[2])
    print("solving----------------------")
    print(scalars)

    return(scalars)


x_base = np.array((0.0, 1.0,0.5, 0.0,-0.5,0.0,0.0,-2.5)).reshape((4,2))
x_frame = np.array((0.0, 0.9,0.46, 0.0,-0.49,-0.0,0.0,-2.5)).reshape((4,2))

print(x_base)
print(x_frame)


x_frame_3d = get_z_coords(x_frame, x_base)
print(x_frame_3d)

x_trial = x_frame_3d
basis = get_ON_basis(x_trial)

#just do quick rotation to test
rot1 = np.zeros((3,3))
theta = 1.0
rot1[0,0] = np.cos(theta)
rot1[0,1] = -np.sin(theta)
rot1[1,0] = np.sin(theta)
rot1[1,1] = np.cos(theta)
rot1[2,2] = 1.0

x_trial_2 = rot1 @ x_frame_3d.T
x_trial_2_n = x_trial_2.T

print("trial1--------------------------")
print(x_trial)
print("trial2-----------------------------")
print(x_trial_2_n)

basis2 = get_ON_basis(x_trial_2_n)

scalars = get_linear_combination(basis, basis2)


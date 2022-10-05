import numpy as np

"""
NOTES: 
x0 is 3 length values:
    a0: tail to midpt at back of head (midpt between ears), 
    b0: ear to ear,
    c0: midpt back of head to snout
    
ASSUME:
    tail point has z=0 
"""

def get_lengths(x):
    #midpoint at back of head
    midpt = (x[1] + x[2]) / 2 #coordinate

    #define 3 axes of interest corresponding to x0
    body_axis = midpt - x[0] #body vec
    ear_axis = x[2] - x[1]
    head_axis = x[3] - midpt #gaze vec in 2d

    #get lengths
    ba_len = np.linalg.norm(body_axis)
    ea_len = np.linalg.norm(ear_axis)
    ha_len = np.linalg.norm(head_axis)

    return ba_len , ea_len , ha_len

def three_dim_values(x, x0):
    """
    :param x: 4 input coords in 2d-> ( tail, left ear, right ear, snout)
    :param x0: as above
    :return: 4 coords in 3d
    """

    # Get length of head and body vectors
    ba_len , ea_len , ha_len = get_lengths(x)

    #get verticals along three axes
    ba_height = np.sqrt(x0[0]**2 - ba_len**2)
    ea_height = np.sqrt(x0[1]**2 - ea_len**2)
    ha_height = np.sqrt(x0[2]**2 - ha_len**2)

    """
    if np.isnan(ba_height):
        print("ba_height: ", ba_height)
        print("x0[0]**2 = ", x0[0]**2)
        print("ba_len**2 = ", ba_len**2)

    
    if np.isnan(ea_height):
        print("ea_height: ", ea_height)
        print("x0[1]**2 = ", x0[1]**2)
        print("ba_len**2 = ", ea_len**2)

        
    if np.isnan(ha_height):
        print("ha_height: ", ha_height)
        print("x0[2]**2 = ", x0[2]**2)
        print("ha_len**2 = ", ha_len**2)
    """

    #get z coords of each orig point
    #note these are arbitrarily set: z0 (tail) = 0, z1 < z2 etc, we may be able to add constraints to these!!
    z0 = 0.0
    z1 = ba_height - ea_height/2
    z2 = ba_height + ea_height/2
    z3 = ba_height + ha_height

    #1: points in 3d
    a = np.append(x[0], z0)
    b = np.append(x[1], z1)
    c = np.append(x[2], z2)
    d = np.append(x[3], z3)

    x_3d = np.concatenate((a, b, c, d), axis=0).reshape(4, 3)

    return x_3d

def gaze_vec(x):
    """
    :param x: 4 input points in 3d
    :return: gaze vector (middle of head to nose) in 3d
    """
    midpt = (x[1] + x[2]) / 2
    g_vec = x[3] - midpt

    return g_vec

def pos_vec(x):
    """

    :param x: as before (3d)
    :return: position of centre of gravity of mouse within arena
    """
    midpt = (x[1]+x[2]) / 2
    cog = (midpt + x[0]) /2

    return cog

def pitch(ha_len,c0):
    """
    c0: length midpt back of head to snout - baseline
    ha_len: same as above but current observation
    """

    pitch = np.arccos(c0/ha_len)

    return pitch

def roll(ea_len,b0):
    """
    b0: length ear to ear - baseline
    ha_len: same as above but current observation
    """

    roll = np.arccos(b0/ea_len)

    return roll

def h_angle_x(g_vec):
    """
    :param p_vec: pose vec
    :return: angle of head relative to "X" axis

    """

    head_angle = np.arccos(g_vec[0]/(np.sqrt(g_vec[0]**2 + g_vec[1]**2)))
    return head_angle

def body_vec(x):
    """

    :param x: cool again 3d
    :return: vector specifying body length / direction from tail to back of head between ears
    """
    midpt = (x[1] + x[2]) / 2
    b_vec = midpt - x[0]

    return b_vec

def b_angle_x(b_vec):
    """
    :param b_vec: body orientation vector
    :return: body angle w respect to "X" axis
    """
    body_angle = np.arccos(b_vec[0]/(np.sqrt(b_vec[0]**2 + b_vec[1]**2)))

    return body_angle

def h_b_angle(head_angle, body_angle):
    """
    :param head_angle:
    :param body_angle:
    :return: relative angle of head to the body, suggests if head is turned to one side
    """
    head_body_angle = head_angle - body_angle

    return head_body_angle

def get_output_variables(x, x0):
    x_3d = three_dim_values(x, x0)
    g_vec = gaze_vec(x_3d)
    cog = pos_vec(x_3d)
    b_vec = body_vec(x_3d)
    head_angle = h_angle_x(g_vec)
    body_angle = b_angle_x(b_vec)
    head_body_angle = h_b_angle(head_angle, body_angle)

    ba_len , ea_len , ha_len = get_lengths(x)
    pitch_angle = pitch(ha_len, x0[2])
    roll_angle = roll(ea_len, x0[1])

    variables_list = [x_3d, g_vec, cog, b_vec, head_angle, body_angle, pitch_angle, roll_angle, head_body_angle]

    return variables_list



def main():
    x0 = np.array((4.74, 3.16, 1.58))
    x = np.array(((0.0,0.0), (4.8,0.0), (3.6,2.7), (5.4,1.6)))

    import time
    t_start = time.time()
    print(get_output_variables(x, x0))
    print(time.time() - t_start)


if __name__ == "__main__":
    main()











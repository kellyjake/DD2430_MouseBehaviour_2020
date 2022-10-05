import numpy as np

"""
NOTES:
x is 4x2:
    snout, le, re, tail
    ===> so need changes in this and pose.py i think

x0 is 3 length values:
    a0: tail to midpt at back of head (midpt between ears),
    b0: ear to ear,
    c0: midpt back of head to snout
    -----------------------------------------
    BASELINE NOT NEEDED FOR 2D analysis!!!!!!

ASSUME:

"""


def get_midpt(x):
    midpt = (x[1] + x[2]) / 2
    return midpt

def get_lengths(x):
    # midpoint at back of head
    midpt = (x[1] + x[2]) / 2  # coordinate

    # define 3 axes of interest corresponding to x0
    body_axis = midpt - x[3]  # body vec
    ear_axis = x[2] - x[1]
    head_axis = x[0] - midpt  # gaze vec in 2d

    # get lengths
    ba_len = np.linalg.norm(body_axis)
    ea_len = np.linalg.norm(ear_axis)
    ha_len = np.linalg.norm(head_axis)

    return ba_len, ea_len, ha_len


def gaze_vec(x):
    """
    :param x: 4 input points in 3d
    :return: gaze vector (middle of head to nose) in 3d
    """
    midpt = (x[1] + x[2]) / 2
    g_vec = x[0] - midpt

    return g_vec


def pos_vec(x):
    """

    :param x: as before (3d)
    :return: position of centre of gravity of mouse within arena
    """
    midpt = (x[1] + x[2]) / 2
    cog = (midpt + x[3]) / 2

    return cog


def h_angle_x(g_vec):
    """
    :param p_vec: pose vec
    :return: angle of head relative to "X" axis

    """

    head_angle = np.sign(g_vec[1]) * np.arccos(g_vec[0] / (np.sqrt(g_vec[0] ** 2 + g_vec[1] ** 2)))
    return head_angle


def body_vec(x):
    """

    :param x: cool again 3d
    :return: vector specifying body length / direction from tail to back of head between ears
    """
    midpt = (x[1] + x[2]) / 2
    b_vec = midpt - x[3]

    return b_vec


def b_angle_x(b_vec):
    """
    :param b_vec: body orientation vector
    :return: body angle w respect to "X" axis
    """
    body_angle = np.sign(b_vec[1]) * np.arccos(b_vec[0] / (np.sqrt(b_vec[0] ** 2 + b_vec[1] ** 2)))

    return body_angle


def h_b_angle(head_angle, body_angle):
    """
    :param head_angle:
    :param body_angle:
    :return: relative angle of head to the body, suggests if head is turned to one side
    """
    hb_ang = head_angle - body_angle
    head_body_angle = (hb_ang + 2*np.pi) % (-np.sign(hb_ang - np.pi * np.sign(hb_ang)) * np.pi)

    return head_body_angle


def get_output_variables_2d(x):
    ba_len, ea_len, ha_len = get_lengths(x)
    g_vec = gaze_vec(x)
    cog = pos_vec(x)
    b_vec = body_vec(x)
    head_angle = h_angle_x(g_vec)
    body_angle = b_angle_x(b_vec)
    head_body_angle = h_b_angle(head_angle, body_angle)

    variables_list = [ba_len, g_vec, cog, b_vec, head_angle, body_angle, head_body_angle]

    return variables_list

"""
h1 = 120
b = 20
h2 = 300
h3 = -120
h4 =-300


hb1 = h_b_angle(h1,b)
hb2 = h_b_angle(h2,b)
hb3 = h_b_angle(h3,b)
hb4 = h_b_angle(h4,b)
print(hb1,hb2,hb3,hb4)
"""
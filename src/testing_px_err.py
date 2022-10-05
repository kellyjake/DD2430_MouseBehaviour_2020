import numpy as np
from pose2d import get_output_variables_2d

screen = np.array([15,15])
img = np.array([448,448])

snout = [9.4,1.5]
lear = [9.2,1.9]
rear = [9.6,1.8]
tail = [10,3.1]



invec = np.array([snout,lear,rear,tail])  /  (15/448)
invec_err = invec.copy()
mpe = 0.15
noise = np.random.normal(0,mpe,(4,2))
invec_err += noise

print(invec)
print(invec_err)

#ba_len , g_vec , cog , b_vec , head_angle , body_angle , head_body_angle
ret = get_output_variables_2d(invec)
ret_err = get_output_variables_2d(invec_err)

ba_len_err , g_vec_err , cog_err , b_vec_err , head_angle_err , body_angle_err , head_body_angle_err = np.abs(np.array(ret) - np.array(ret_err))

print(f'Error when adding zero mean {mpe} sigma random noise')
print(f'Random noise: {noise}')

print(f'ba_len_err : {ba_len_err} pixels')
print(f'g_vec_err : {g_vec_err} pixels')
print(f'cog_err : {cog_err} pixels')
print(f'b_vec_err : {b_vec_err} pixels')
print(f'head_angle_err : {np.rad2deg(head_angle_err)} degrees')
print(f'body_angle_err : {np.rad2deg(body_angle_err)} degrees')
print(f'head_body_angle_err : {np.rad2deg(head_body_angle_err)} degrees')
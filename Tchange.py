import numpy as np
import random

A = 7.9
B = 6.06 
gamma = 85.8
radian = gamma / 180 * 3.14159

T1 = 400
T2 = 500
T3 = 600
Atoms = 100
eV_2_kcal_mol = 23.061
kB = 8.617E-5 * eV_2_kcal_mol

P_2_Q = np.array([-4, 2.5])

Q_in = np.array([[True,  True,  0,  1,  1.71,  0.0811,  0],
				 [True,  True,  0, -1,  1.71,  0.0811,  0],
				 [True,  True, -1,  1,  1.29,  0.0375,  0],
				 [True,  True,  1, -1,  1.29,  0.0375,  0],
				 [True,  True, -1,  0,  1.30,  0.00144,  0],
				 [True,  True,  1,  0,  1.30,  0.00144,  0],
				 [True,  False,  0, -1,  3.31,  0.104,  0],
				 [True,  False,  0,  1,  11.9,  0.581,  0],
				 [False,  False, -1,  0,  4.11E-2, -2.16E-4,  3.78E-7],
				 [False,  False, -1,  1,  2.67E-1, -1.14E-3,  1.31E-6],
				 [False,  False,  0,  0, -5.71E-2,  1.16E-3, -1.62E-6],
				 [False,  False, -1,  2,  1.71E-3, -1.16E-5,  1.98E-8]])

P_in = np.array([[True, False,  0,  1,  1.59,  0.103,  0],
				 [True, False,  0, -1,  1.59,  0.103,  0],
				 [True, False, -1,  1,  2.42,  0.0257, 0],
				 [True, False,  1, -1,  2.42,  0.0257, 0],
				 [True, False, -1,  0,  2.21,  0.0145, 0],
				 [True, False,  1,  0,  2.21,  0.0145, 0],
				 [True, True,  0, -1,  11.9,  0.581,  0],
				 [True, True,  0,  1,  3.31,  0.104,  0],
				 [False, True,  1, -2,  1.71E-3, -1.16E-5,  1.98E-8],
				 [False, True,  0,  0, -5.71E-2,  1.16E-3, -1.62E-6],
				 [False, True,  1, -1,  2.67E-1, -1.14E-3,  1.31E-6],
 				 [False, True,  1,  0,  4.11E-2, -2.16E-4,  3.78E-7]])

def Arr(E, A, T = T1, k = kB):
	return  A * np.exp(-E / (k * T))

def Poly(A, B, C, T = T1):
	return A + B * T + C * (T ** 2)

def iso_D(mean_r_squ, dt, n):
	return mean_r_squ / (2 * dn * t) 

def dt(u, Rn):
	return -np.log([u]) / Rn

r_q = np.zeros((1, 12))
r_p = np.zeros((1, 12))

# Calculate rate of 12 steps for P and Q
for i in range(12):
	if Q_in[i][0]:
		r_q[0][i] = Arr(Q_in[i][4], Q_in[i][5])
	else:
		r_q[0][i] = Poly(Q_in[i][4], Q_in[i][5], Q_in[i][6])

for i in range(12):
	if P_in[i][0]:
		r_p[0][i] = Arr(P_in[i][4], P_in[i][5])
	else:
		r_p[0][i] = Poly(P_in[i][4], P_in[i][5], P_in[i][6])

# Calculate total rate for 1 atom
for i in range(1, 12):
	r_p[0][i] = r_p[0][i - 1] + r_p[0][i]
	r_q[0][i] = r_q[0][i - 1] + r_q[0][i]

q_minus_p = r_q[0][-1] - r_p[0][-1]

R_n = np.zeros(Atoms * 12 + 1)

for i in range(Atoms):
	for j in range(12):
		R_n[i * 12 + j + 1] = r_q[0][-1] * i + r_q[0][j]


# Multiply equal to 50,000,000
sampling_times = 10000
sampling_size = 5000

# Assume start from P point
At_Q = True
step_location = np.zeros((Atoms, 2))
delta_t = np.zeros(sampling_times)
PQ_check = np.ones((Atoms, 1), dtype=bool)

# x and y displacement
x = np.zeros((Atoms, 1))
y = np.zeros((Atoms, 1))

datax = np.zeros((sampling_times, Atoms))
datay = np.zeros((sampling_times, Atoms))

for i in range(sampling_times):
	for j in range(sampling_size):
		
		u_r = random.random()
		u_t = random.random()

		R_i = u_r * R_n[-1]
		R_temp = np.sort(np.hstack([R_n, [R_i]]))
		n_th = R_temp.tolist().index(R_i) - 1

		atom_num = int(n_th / 12)  # 0 - 99

		step_num = n_th % 12      # 0 - 11	

		At_Q = np.copy(PQ_check[atom_num])

		if At_Q:
			Pts_in = Q_in
		else:
			Pts_in = P_in

		PQ_check[atom_num] = np.copy(Pts_in[step_num][1])
		step_location[atom_num] += np.array([Pts_in[step_num][2], Pts_in[step_num][3]])
		delta_t[i] += dt(u_t, R_n[-1])
		# Calculate new Rn

		# ########### Method 3 ###########
		if At_Q:
			if not PQ_check[atom_num]:
				for m in range(12):
					R_n[atom_num * 12 + m + 1] = np.copy(R_n[atom_num * 12] + r_q[0][m])
				for n in range((atom_num + 1) * 12 + 1, Atoms * 12 + 1):
					R_n[n] -= q_minus_p 
		else:
			if PQ_check[atom_num]:
				for m in range(12):
					R_n[atom_num * 12 + m + 1] = np.copy(R_n[atom_num * 12] + r_p[0][m])
				for n in range((atom_num + 1) * 12 + 1, Atoms * 12 + 1):
					R_n[n] += q_minus_p 
		################################

	for j in range(Atoms):
		x[j], y[j] = step_location[j][0] * A, step_location[j][1] * B
		if not PQ_check[j]:
			x[j] += P_2_Q[0]
			y[j] += P_2_Q[1]

	for j in range(Atoms):
		datax[i][j] = np.copy(x[j])
		datay[i][j] = np.copy(y[j])

np.savetxt('solux.csv', datax, delimiter=",")
np.savetxt('soluy.csv', datay, delimiter=",")
np.savetxt('dt.csv', delta_t, delimiter=",")
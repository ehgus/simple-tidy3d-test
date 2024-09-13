""" Adjoint method for diffraction control.
    Currently, adjoint sources are implemented using plane wave sources.
    Auxiliary filters and phasor-based FoM are implemented.
"""
import time, os, sys
import numpy as np
import pandas as pd
from scipy import io

import tidy3d as td
from tidy3d import web
from tidy3d import SpatialDataArray

current_abs_path = os.path.abspath('')

def calculate_angles(k0, period, target):
    num_order = len(target)
    angles = np.zeros(num_order, dtype=float)
    dkx = 2*np.pi/period
    for ind, diff_order in enumerate(target.keys()):
        angles[ind] = np.arcsin(diff_order * dkx / k0)
    return angles

def cal_overlap_int(E, H, target, wl, period, impedance):
    """ This function calculates and returns overlap integral of each mode and the fields"""
    sin_theta = np.array(list(target.keys())).reshape(7,1)*wl/period
    cos_theta = (1-sin_theta**2)**(0.5)

    coord_length = E.shape[1]
    num_order = len(target)
    phase_factors = np.zeros(shape=(num_order, coord_length), dtype=complex)
    
    for ind, diff_order in enumerate(target.keys()):
        phase_factors[ind] = np.exp(1j*diff_order * np.linspace(-np.pi, np.pi, coord_length+1))[:-1]
    phase_factors = np.array(phase_factors, dtype=complex)
    
    shape = (num_order, coord_length, 3)
    Em_plus = np.zeros(shape=shape, dtype=complex)
    Hm_plus = np.zeros(shape=shape, dtype=complex)
    Em_plus[...,2] = phase_factors
    Hm_plus[...,0] = - phase_factors / impedance * cos_theta
    Hm_plus[...,1] = - phase_factors / impedance * sin_theta
    Em_minus = np.conj(Em_plus)
    Hm_minus = np.conj(Hm_plus)
    
    normal_integrand = (np.cross(Em_plus, Hm_minus) + np.cross(Em_minus, Hm_plus))[...,1]
    normal_integral = np.abs(np.trapz(normal_integrand, dx=period))
    
    integrand = (np.cross(E,Hm_minus) + np.cross(Em_minus, H))[...,1]
    integral = np.trapz(integrand, dx=period, axis=1)
    
    return integral, normal_integral

def calculate_factor(E, H, target, wl, period, impedance):
    """ This function returns some additional factors in chain rules.
    Since most of the functionalities are similar in calculate_fom function,
    it needs to be modified in more abstract manner"""
    
    target_vals = np.array(list(target.values()))
    integral, normal_integral = cal_overlap_int(E, H, target, wl, period, impedance)
 
    efficiency = abs(integral / normal_integral)**2
    t_conj = np.conj(integral)/abs(normal_integral)
    
    factor = (efficiency-target_vals)
    
    return factor, t_conj
    
def calculate_fom(E, H, target, wl, period, impedance):
    """ Given field monitor values, one can calulate diffraction efficiency.
        Manually calculate script function 'grating' """
    target_vals = np.array(list(target.values()))
    integral, normal_integral = cal_overlap_int(E, H, target, wl, period, impedance)
    intensity = abs(integral/normal_integral)**2
    print(f'Diffraction intensity = {intensity}')
    print(f'target = {target_vals}')
    return np.sum(abs(intensity-target_vals)**2, axis=-1)

def generate_nk(rho, n_max, n_min, gridx, gridy, gridz):
    rho = np.insert(rho, len(rho), rho[0])
    num = len(rho)
    RI = n_min + (n_max - n_min) * rho
    RI = RI.reshape(len(gridx),len(gridy),len(gridz))
    n_dataset = SpatialDataArray(RI, coords=dict(x=gridx, y=gridy, z=gridz))
    mat_custom = td.CustomMedium.from_nk(n_dataset, interp_method="nearest")
    return mat_custom

def forward_sim(sim_params, sim_name, target, wl, period, impedance, freq0, Efield_norm_value):
    nm = 1e-3
    sim_params['sources'] = [td.PlaneWave(
        center=(0, 500*nm, 0),
        direction="-",
        size=(td.inf, 0, td.inf),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 20, amplitude = 1, phase = 0),
        pol_angle=np.pi / 2,
    )]
    
    sim = td.Simulation(**sim_params)
    sim_data_file = f"{current_abs_path}/data/forward_{sim_name}.hdf5"
    if os.path.isfile(sim_data_file):
        sim_data = td.SimulationData.from_file(fname=sim_data_file)
    else:
        sim_data = web.run(
            sim,
            task_name=f"forward_{sim_name}",
            path=sim_data_file,
            verbose=False
        )
    E_fwd = sim_data.load_field_monitor('design monitor').Ez.squeeze()[:-1]/Efield_norm_value
    field_monitor_rst = sim_data.load_field_monitor('field monitor')    
    E = np.dstack((field_monitor_rst.Ex.squeeze(),
                  field_monitor_rst.Ey.squeeze(),
                  field_monitor_rst.Ez.squeeze()
                 ))[:,:-1,:]/Efield_norm_value
    H = np.dstack((field_monitor_rst.Hx.squeeze(),
                  field_monitor_rst.Hy.squeeze(),
                  field_monitor_rst.Hz.squeeze()
                 ))[:,:-1,:]/Efield_norm_value
    
    fom = calculate_fom(E, H, target, wl, period, impedance)
    factors, t_conj = calculate_factor(E, H, target, wl, period, impedance)
    
    return fom, E, H, E_fwd, factors, t_conj

def adjoint_sim(sim_params, sim_name, factors, t_conj, angles, freq0, Efield_norm_value):
    nm = 1e-3
    amps = np.abs(factors)
    phases = np.angle(factors * t_conj) + np.pi/2  

    sim_params['sources'] = []
    for i in range(len(factors)):
        adjoint_source = td.PlaneWave(
            center=(0, -500*nm, 0),
            direction="+",
            size=(td.inf, 0, td.inf),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 20, amplitude = amps[i], phase = phases[i]),
            angle_theta=angles[i],
            pol_angle=np.pi / 2,
        )
        sim_params['sources'].append(adjoint_source)
    sim = td.Simulation(**sim_params)
    sim_data_file = f"{current_abs_path}/data/adjoint_{sim_name}.hdf5"
    if os.path.isfile(sim_data_file):
        sim_data = td.SimulationData.from_file(fname=sim_data_file)
    else:
        sim_data = web.run(
            sim,
            task_name=f"adjoint_{sim_name}",
            path=sim_data_file,
            verbose=False
        )
    E_adj = sim_data.load_field_monitor('design monitor').Ez.squeeze()[:-1]/Efield_norm_value
        
    return E_adj

def calculate_grad(E_fwd, E_adj, rho, n_max, n_min):
    # grad_RI = E_fwd*E_adj*n=E_fwd*E_adj*(n_min+dn*rho) 
    # grad_density = real(E_fwd*E_adj*n/dn)
    #              = real(E_fwd*E_adj*(n_min/dn+rho))
    gradient = E_fwd * E_adj
    gradient = np.mean(gradient,axis=1)
    gradient *= 2*(n_min/(n_max - n_min) + rho)
    gradient = np.real(gradient)
    return gradient  
 
def update_design(rho, gradient, n_max, n_min, step_size=0.001, maximization=True):
    if maximization:
        rho = rho + step_size * gradient
    else:
        rho = rho - step_size * gradient
    for i in range(len(rho)-1):
        if rho[i] > 1.0:
            rho[i] = 1.0
        elif rho[i] < 0.:
            rho[i] = 0.
    return rho

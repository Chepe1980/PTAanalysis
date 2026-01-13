import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
import io
import base64

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# Set page configuration
st.set_page_config(
    page_title="Austin Chalk Pressure Transient Analysis",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# 1. SMOOTH PRESSURE TRANSIENT SIMULATOR (IMPROVED)
# ============================================================================

class AustinChalkPressureTransient:
    """Simulate pressure transient behavior for horizontal wells in Austin Chalk"""

    def __init__(self):
        # Reservoir parameters based on paper
        self.k = 5.0  # Permeability in md (0.5-10 md range from paper)
        self.h = 200.0  # Net thickness in ft (100-400 ft from paper)
        self.phi = 0.08  # Porosity (less than 8% from other paper)
        self.ct = 1e-5  # Total compressibility, 1/psi
        self.mu = 1.0  # Viscosity, cp
        self.B = 1.2  # Formation volume factor
        self.rw = 0.25  # Wellbore radius, ft
        self.Lw = 2000.0  # Horizontal well length, ft
        self.omega = 0.05  # Storativity ratio (0.01-0.1 from paper)
        self.lambda_val = 1e-7  # Interporosity flow coefficient (~1e-7 from paper)

        # More points for smoother curves
        self.t_early = np.logspace(-4, -1, 200)
        self.t_mid = np.logspace(-1, 1, 300)
        self.t_late = np.logspace(1, 3, 200)

        # Concatenate and then get unique sorted values to ensure strictly increasing sequence
        self.time = np.unique(np.concatenate([self.t_early, self.t_mid, self.t_late]))

    def smooth_wellbore_storage_response(self, t):
        """Smooth early time response dominated by wellbore storage"""
        C = 100.0  # Wellbore storage coefficient, bbl/psi (large as noted in paper)
        q = 1000.0  # Flow rate, STB/D
        B = self.B

        # Unit slope with smooth transition
        delta_p = (q * B / (24 * C)) * t

        # Add smooth transition effect
        transition_time = 0.01
        transition_factor = 1 / (1 + np.exp(-10*(t - transition_time)/transition_time))
        delta_p *= (1 + 0.5*transition_factor)  # Smooth increase

        derivative = delta_p  # Unit slope means derivative equals pressure change

        return delta_p, derivative

    def smooth_radial_flow_response(self, t):
        """Smooth early radial flow in fracture planes"""
        q = 1000.0
        k = self.k
        h = self.h
        mu = self.mu
        B = self.B
        phi = self.phi
        ct = self.ct
        rw = self.rw

        # Radial flow equation with smoothing
        t_safe = np.maximum(t, 1e-10)  # Avoid log(0)
        delta_p = (162.6 * q * B * mu / (k * h)) * (
            np.log10(t_safe) + np.log10(k / (phi * mu * ct * rw**2)) - 3.23 + 0.87 * 0
        )

        # Smooth derivative for radial flow
        derivative = 70.6 * q * B * mu / (k * h) * np.ones_like(t)

        # Smooth transition into radial flow
        transition = 1 / (1 + np.exp(-100*(t - 0.005)))
        derivative = derivative * transition

        return delta_p, derivative

    def smooth_linear_flow_response(self, t):
        """Smooth linear flow regime characteristic of horizontal wells"""
        q = 1000.0
        k = self.k
        h = self.h
        mu = self.mu
        B = self.B
        phi = self.phi
        ct = self.ct
        Lw = self.Lw

        # Linear flow equation: delta_p ~ sqrt(t) with smoothing
        t_safe = np.maximum(t, 1e-10)
        delta_p = (8.128 * q * B / (Lw * h)) * np.sqrt(mu * t_safe / (phi * ct * k))

        # Add smooth onset
        linear_start = 0.05
        linear_factor = 1 / (1 + np.exp(-50*(t - linear_start)/linear_start))
        delta_p *= linear_factor

        # Derivative for linear flow
        derivative = delta_p / 2.0

        return delta_p, derivative

    def smooth_dual_porosity_transition(self, t, pre_transition_pressure, pre_transition_derivative):
        """Smooth dual porosity transition with pseudosteady state interporosity flow"""
        omega = self.omega

        # Create a smooth "valley" in the derivative
        transition_start = 1.0
        transition_end = 10.0

        derivative = pre_transition_derivative.copy()
        pressure = pre_transition_pressure.copy()

        # Find indices in transition zone
        transition_mask = (t >= transition_start) & (t <= transition_end)

        if np.any(transition_mask):
            # Create smooth dip in derivative using Gaussian
            t_transition = t[transition_mask]
            t_norm = (t_transition - transition_start) / (transition_end - transition_start)

            # Depth of dip depends on omega (smaller omega = deeper dip)
            dip_depth = 0.3 * (1 - omega/0.1)

            # Gaussian dip for smooth transition
            dip_factor = 1 - dip_depth * np.exp(-((t_norm - 0.5)**2) / (0.2))
            dip_factor = np.clip(dip_factor, 0.3, 1.0)

            derivative[transition_mask] *= dip_factor

            # Pressure response flattens during transition
            pressure[transition_mask] = pressure[transition_mask] * (0.9 + 0.1 * t_norm)

        return pressure, derivative

    def generate_very_smooth_data(self, well_type="typical"):
        """Generate very smooth synthetic pressure transient data"""

        # Generate base responses
        delta_p_total = np.zeros_like(self.time)
        derivative_total = np.zeros_like(self.time)

        if well_type == "typical":  # Well A, D
            # Strong linear flow with dual porosity transition
            for i, t in enumerate(self.time):
                if t < 0.01:
                    dp, deriv = self.smooth_wellbore_storage_response(t)
                elif t < 0.1:
                    dp, deriv = self.smooth_radial_flow_response(t)
                elif t < 100:
                    dp, deriv = self.smooth_linear_flow_response(t)
                else:
                    dp, deriv = self.smooth_linear_flow_response(t)
                    dp *= 1.2  # Late time adjustment
                    deriv *= 1.2

                delta_p_total[i] = dp
                derivative_total[i] = deriv

            # Apply smooth dual porosity transition
            delta_p_total, derivative_total = self.smooth_dual_porosity_transition(
                self.time, delta_p_total, derivative_total
            )

        elif well_type == "damaged":  # Well B
            # Shows radial flow with high skin
            skin_factor = 13.0
            for i, t in enumerate(self.time):
                if t < 0.1:
                    dp, deriv = self.smooth_wellbore_storage_response(t)
                    # Add skin effect smoothly
                    skin_effect = skin_factor * (1 - np.exp(-t/0.01))
                    dp += 100 * skin_effect
                elif t < 1:
                    dp, deriv = self.smooth_radial_flow_response(t)
                    dp += 100  # High skin
                else:
                    dp, deriv = self.smooth_linear_flow_response(t)

                delta_p_total[i] = dp
                derivative_total[i] = deriv

        elif well_type == "good_producer":  # Well D with clear dual porosity
            # Clear dual porosity transition
            self.omega = 0.02  # Smaller for clearer transition

            for i, t in enumerate(self.time):
                if t < 0.001:
                    dp, deriv = self.smooth_wellbore_storage_response(t)
                elif t < 0.01:
                    dp, deriv = self.smooth_radial_flow_response(t)
                elif t < 10:
                    dp, deriv = self.smooth_linear_flow_response(t)
                else:
                    dp, deriv = self.smooth_linear_flow_response(t)

                delta_p_total[i] = dp
                derivative_total[i] = deriv

            # Strong dual porosity transition
            delta_p_total, derivative_total = self.smooth_dual_porosity_transition(
                self.time, delta_p_total, derivative_total
            )

        # Apply cubic spline smoothing for ultra-smooth curves
        log_time = np.log10(self.time)
        spline_p = CubicSpline(log_time, delta_p_total)
        spline_deriv = CubicSpline(log_time, derivative_total)

        # Evaluate on finer grid
        fine_log_time = np.linspace(log_time[0], log_time[-1], 1000)
        fine_time = 10**fine_log_time
        fine_delta_p = spline_p(fine_log_time)
        fine_derivative = spline_deriv(fine_log_time)

        # Minimal noise for realism
        noise_level = 0.005  # Very small noise
        fine_delta_p *= (1 + noise_level * np.random.randn(len(fine_delta_p)))
        fine_derivative *= (1 + noise_level * np.random.randn(len(fine_derivative)))

        # Ensure positivity
        fine_delta_p = np.maximum(fine_delta_p, 0.1)
        fine_derivative = np.maximum(fine_derivative, 0.1)

        return fine_time, fine_delta_p, fine_derivative

# ============================================================================
# 2. PLOTTING FUNCTIONS
# ============================================================================

def create_diagnostic_log_log_plot(well_name, time, delta_p, derivative, fig_num, real_data=None):
    """Create log-log diagnostic plot (Figures 7, 11, 15, 18 in paper)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot pressure change and derivative
    ax.loglog(time, delta_p, 'bo-', markersize=4, linewidth=1.5,
              label='Synthetic ΔP (psi)', alpha=0.7)
    ax.loglog(time, derivative, 'rs-', markersize=4, linewidth=1.5,
              label='Synthetic Derivative (psi)', alpha=0.7)
    
    # Plot real data if available
    if real_data is not None and 'Time' in real_data.columns:
        if 'Delta_P' in real_data.columns:
            ax.loglog(real_data['Time'], real_data['Delta_P'], 'go-', markersize=6, linewidth=2,
                     label='Real ΔP (psi)', alpha=0.8)
        if 'Derivative' in real_data.columns:
            ax.loglog(real_data['Time'], real_data['Derivative'], 'mo-', markersize=6, linewidth=2,
                     label='Real Derivative (psi)', alpha=0.8)

    # Add reference lines for different flow regimes
    # Unit slope for wellbore storage
    unit_slope_x = np.array([time[0], time[-1]])
    unit_slope_y = 10 * (unit_slope_x / unit_slope_x[0])
    ax.loglog(unit_slope_x, unit_slope_y, 'k--', linewidth=1, alpha=0.5,
              label='Unit Slope (Wellbore Storage)')

    # Half slope for linear flow
    half_slope_x = np.array([1e-1, 1e2])
    half_slope_y = 100 * np.sqrt(half_slope_x / half_slope_x[0])
    ax.loglog(half_slope_x, half_slope_y, 'g--', linewidth=1, alpha=0.5,
              label='½ Slope (Linear Flow)')

    # Constant derivative for radial flow
    radial_level = np.mean(derivative[(time > 0.01) & (time < 0.1)])
    if radial_level > 0:
        ax.axhline(y=radial_level, color='m', linestyle='--', alpha=0.5,
                   label='Radial Flow Level')

    # Formatting
    ax.set_xlabel('Elapsed Time, Δt (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure Change and Derivative (psi)', fontsize=12, fontweight='bold')
    title = f'Figure {fig_num}: Log-Log Diagnostic Plot - {well_name}'
    if real_data is not None:
        title += ' (Synthetic + Real Data)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Add text annotations for flow regimes
    ax.text(0.005, 1, 'Wellbore\nStorage', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    if 'damaged' not in well_name.lower():
        ax.text(0.5, 10, 'Linear Flow', fontsize=9, rotation=26,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))

        # Mark dual porosity transition if visible
        if np.min(derivative) < 0.5 * np.max(derivative[(time > 0.1) & (time < 10)]):
            ax.text(5, np.min(derivative)*0.8, 'Dual Porosity\nTransition',
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))

    plt.tight_layout()
    return fig

def create_sqrt_time_plot(well_name, time, delta_p, fig_num, real_data=None):
    """Create square root of time plot (Figures 8, 12, 16, 19 in paper)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use equivalent time for buildup
    t_equiv = time  # Simplified for demonstration

    # Plot pressure vs sqrt(time)
    sqrt_time = np.sqrt(t_equiv)
    ax.plot(sqrt_time, delta_p, 'bo-', markersize=4, linewidth=1.5, alpha=0.7, label='Synthetic Data')
    
    # Plot real data if available
    if real_data is not None and 'Time' in real_data.columns and 'Delta_P' in real_data.columns:
        real_sqrt_time = np.sqrt(real_data['Time'])
        ax.plot(real_sqrt_time, real_data['Delta_P'], 'go-', markersize=6, linewidth=2, 
                alpha=0.8, label='Real Data')

    # Identify linear flow segments
    # Early linear flow (pre-transition)
    mask_early = (t_equiv > 0.1) & (t_equiv < 1.0)
    if np.sum(mask_early) > 5:
        coeff_early = np.polyfit(sqrt_time[mask_early], delta_p[mask_early], 1)
        ax.plot(sqrt_time[mask_early],
                np.polyval(coeff_early, sqrt_time[mask_early]),
                'r-', linewidth=2, label='Early Linear Flow')

        # Calculate slope m1
        m1 = coeff_early[0]
        ax.text(np.mean(sqrt_time[mask_early]),
                np.mean(delta_p[mask_early]),
                f'm₁ = {m1:.1f}', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

    # Late linear flow (post-transition)
    mask_late = (t_equiv > 10) & (t_equiv < 100)
    if np.sum(mask_late) > 5:
        coeff_late = np.polyfit(sqrt_time[mask_late], delta_p[mask_late], 1)
        ax.plot(sqrt_time[mask_late],
                np.polyval(coeff_late, sqrt_time[mask_late]),
                'g-', linewidth=2, label='Late Linear Flow')

        # Calculate slope m2
        m2 = coeff_late[0]
        ax.text(np.mean(sqrt_time[mask_late]),
                np.mean(delta_p[mask_late]),
                f'm₂ = {m2:.1f}', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))

        # Calculate storativity ratio ω = m1/m2
        if 'm1' in locals():
            omega_calc = (m1/m2)**2
            ax.text(0.05, 0.95, f'ω ≈ {omega_calc:.3f}',
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

    ax.set_xlabel('√(Equivalent Time) (√hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure Change, ΔP (psi)', fontsize=12, fontweight='bold')
    title = f'Figure {fig_num}: Square Root of Time Plot - {well_name}'
    if real_data is not None:
        title += ' (Synthetic + Real Data)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_larsen_plot(well_name, time, derivative, fig_num, real_data=None):
    """Create Larsen plot (Figures 10, 21 in paper)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate sqrt(time)
    sqrt_time = np.sqrt(time)

    # Filter out very early times
    mask = time > 0.001
    sqrt_time_filt = sqrt_time[mask]
    deriv_filt = derivative[mask]

    # Plot derivative vs sqrt(time)
    ax.plot(sqrt_time_filt, deriv_filt, 'bo-', markersize=4, linewidth=1.5, alpha=0.7, label='Synthetic Data')
    
    # Plot real data if available
    if real_data is not None and 'Time' in real_data.columns and 'Derivative' in real_data.columns:
        real_sqrt_time = np.sqrt(real_data['Time'])
        real_mask = real_data['Time'] > 0.001
        ax.plot(real_sqrt_time[real_mask], real_data['Derivative'][real_mask], 
                'go-', markersize=6, linewidth=2, alpha=0.8, label='Real Data')

    # Identify linear flow region
    # In linear flow, derivative vs sqrt(time) should be linear through origin
    mask_linear = (time > 0.1) & (time < 10)
    if np.sum(mask_linear) > 5:
        sqrt_linear = sqrt_time[mask_linear]
        deriv_linear = derivative[mask_linear]

        # Fit line through origin
        slope = np.sum(sqrt_linear * deriv_linear) / np.sum(sqrt_linear**2)

        # Plot fitted line
        x_fit = np.linspace(0, np.max(sqrt_linear), 50)
        y_fit = slope * x_fit
        ax.plot(x_fit, y_fit, 'r--', linewidth=2,
                label=f'Linear Flow Fit: slope = {slope:.2f}')

        # Mark upper limit for radial flow (m*)
        # First point on linear flow line gives m*
        if len(deriv_linear) > 0:
            m_star = deriv_linear[0]
            ax.axhline(y=m_star, color='g', linestyle='--', alpha=0.7,
                      label=f'm* = {m_star:.1f} psi (Radial Flow Limit)')

            # Mark radial flow region
            mask_radial = time < 0.1
            if np.sum(mask_radial) > 3:
                ax.plot(sqrt_time[mask_radial], derivative[mask_radial],
                       'ms', markersize=6, label='Radial Flow Region')

    ax.set_xlabel('√(Elapsed Time) (√hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure Derivative (psi)', fontsize=12, fontweight='bold')
    title = f'Figure {fig_num}: Larsen Plot - {well_name}'
    if real_data is not None:
        title += ' (Synthetic + Real Data)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add explanatory text
    ax.text(0.05, 0.95, 'Larsen Plot Analysis:', transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.90, '1. Radial flow → Constant derivative', transform=ax.transAxes, fontsize=9)
    ax.text(0.05, 0.85, '2. Linear flow → Line through origin', transform=ax.transAxes, fontsize=9)
    ax.text(0.05, 0.80, '3. m* gives lower limit for k', transform=ax.transAxes, fontsize=9)

    plt.tight_layout()
    return fig

def create_type_curve_match(well_name, time, delta_p, derivative, fig_num, real_data=None):
    """Create type-curve match plot (Figures 9, 13, 17, 20 in paper)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Generate theoretical type curves for different ω values
    tD_range = np.logspace(-2, 4, 100)

    # Cinco-Ley & Meng type curves for linear flow in dual porosity
    omega_values = [0.001, 0.01, 0.05, 0.1]
    colors = ['r', 'g', 'b', 'm']

    for omega, color in zip(omega_values, colors):
        # Simplified dual porosity response for linear flow
        # Early response (fracture dominated)
        pD_early = 2 * np.sqrt(tD_range / (np.pi * omega))

        # Transition
        transition_point = -np.log(omega) / 10  # Simplified

        # Late response (total system)
        pD_late = 2 * np.sqrt(tD_range / np.pi)

        # Combine with smooth transition
        weight = 1 / (1 + np.exp(-10*(np.log10(tD_range) - np.log10(transition_point))))
        pD = (1 - weight) * pD_early + weight * pD_late

        # Derivative
        # For linear flow: derivative = 0.5 * pD
        deriv_D = 0.5 * pD

        # Apply dual porosity "dip"
        if omega < 0.1:
            # Create dip in derivative
            dip_center = transition_point
            dip_width = 1.0
            dip_strength = 0.5 * (0.1 - omega) / 0.1

            dip_factor = 1 - dip_strength * np.exp(-((np.log10(tD_range) - np.log10(dip_center))**2) / (2*dip_width**2))
            deriv_D *= dip_factor

        # Plot type curves
        ax1.loglog(tD_range, pD, color=color, linestyle='-', alpha=0.7,
                  linewidth=1.5, label=f'ω = {omega}')
        ax2.loglog(tD_range, deriv_D, color=color, linestyle='-', alpha=0.7,
                  linewidth=1.5, label=f'ω = {omega}')

    # Plot synthetic data (scaled)
    scale_time = 10.0  # Time match factor
    scale_pressure = 100.0  # Pressure match factor

    tD_synth = time * scale_time
    pD_synth = delta_p / scale_pressure
    derivD_synth = np.gradient(pD_synth, np.log(tD_synth))

    ax1.loglog(tD_synth, pD_synth, 'ko-', markersize=4, linewidth=1.5,
              alpha=0.7, label=f'{well_name} Synthetic Data')
    ax2.loglog(tD_synth, derivD_synth, 'ko-', markersize=4, linewidth=1.5,
              alpha=0.7, label=f'{well_name} Synthetic Data')
    
    # Plot real data if available
    if real_data is not None and 'Time' in real_data.columns:
        tD_real = real_data['Time'] * scale_time
        if 'Delta_P' in real_data.columns:
            pD_real = real_data['Delta_P'] / scale_pressure
            ax1.loglog(tD_real, pD_real, 'go-', markersize=6, linewidth=2,
                     alpha=0.8, label=f'{well_name} Real Data')
            
            # Calculate derivative for real data
            if len(real_data) > 1:
                derivD_real = np.gradient(pD_real, np.log(tD_real))
                ax2.loglog(tD_real, derivD_real, 'go-', markersize=6, linewidth=2,
                         alpha=0.8, label=f'{well_name} Real Data')

    # Formatting
    ax1.set_ylabel('Dimensionless Pressure, pD', fontsize=12, fontweight='bold')
    title = f'Figure {fig_num}: Cinco-Ley & Meng Type-Curve Match - {well_name}'
    if real_data is not None:
        title += ' (Synthetic + Real Data)'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)

    ax2.set_xlabel('Dimensionless Time, tD', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dimensionless Derivative', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)

    # Add match information
    match_text = f'Time Match: tD/t = {scale_time:.1f}\nPressure Match: pD/ΔP = {1/scale_pressure:.4f}'
    ax1.text(0.05, 0.95, match_text, transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Indicate estimated ω
    if 'good_producer' in well_name.lower() or 'd' in well_name.lower():
        ax1.text(0.05, 0.85, 'Best Match: ω ≈ 0.01-0.05', transform=ax1.transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

    plt.tight_layout()
    return fig

def create_schematic_plot(simulator, real_data=None):
    """Create schematic flow model plot"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create schematic flow regimes with SMOOTH data
    t_schematic = np.logspace(-3, 3, 1000)

    # Generate smooth theoretical responses
    p_ideal = np.zeros_like(t_schematic)
    deriv_ideal = np.zeros_like(t_schematic)

    for i, t in enumerate(t_schematic):
        if t < 0.01:
            p, d = simulator.smooth_wellbore_storage_response(t)
        elif t < 0.1:
            p, d = simulator.smooth_radial_flow_response(t)
        elif t < 10:
            p, d = simulator.smooth_linear_flow_response(t)
        else:
            p, d = simulator.smooth_linear_flow_response(t)
            p *= 1.2  # Late time adjustment
            d *= 1.2

        p_ideal[i] = p
        deriv_ideal[i] = d

    # Apply smooth dual porosity transition
    p_ideal, deriv_ideal = simulator.smooth_dual_porosity_transition(t_schematic, p_ideal, deriv_ideal)

    # Apply cubic spline for ultra-smooth schematic
    log_t = np.log10(t_schematic)
    spline_p = CubicSpline(log_t, p_ideal)
    spline_d = CubicSpline(log_t, deriv_ideal)

    fine_log_t = np.linspace(log_t[0], log_t[-1], 2000)
    fine_t = 10**fine_log_t
    fine_p = spline_p(fine_log_t)
    fine_d = spline_d(fine_log_t)

    # Plot smooth schematic
    ax.loglog(fine_t, fine_p, 'b-', linewidth=3, alpha=0.7, label='Synthetic ΔP (psi)')
    ax.loglog(fine_t, fine_d, 'r-', linewidth=3, alpha=0.7, label='Synthetic Derivative (psi)')
    
    # Plot real data if available
    if real_data is not None and 'Time' in real_data.columns:
        if 'Delta_P' in real_data.columns:
            ax.loglog(real_data['Time'], real_data['Delta_P'], 'go-', markersize=6, linewidth=2,
                     alpha=0.8, label='Real ΔP (psi)')
        if 'Derivative' in real_data.columns:
            ax.loglog(real_data['Time'], real_data['Derivative'], 'mo-', markersize=6, linewidth=2,
                     alpha=0.8, label='Real Derivative (psi)')

    # Annotate flow regimes
    regimes = [
        (0.001, 1, 'Wellbore\nStorage', 'yellow'),
        (0.02, 10, 'Radial Flow\n(if visible)', 'pink'),
        (0.3, 30, 'Linear Flow', 'lightgreen'),
        (5, 5, 'Dual Porosity\nTransition', 'orange'),
        (50, 80, 'Late Time\nLinear Flow', 'lightblue')
    ]

    for t_pos, p_pos, label, color in regimes:
        ax.text(t_pos, p_pos, label, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.7),
               horizontalalignment='center')

    ax.set_xlabel('Elapsed Time, Δt (hours)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pressure Change and Derivative (psi)', fontsize=14, fontweight='bold')
    title = 'Figure 5: Schematic Pressure Response of Horizontal Wells\nin the Pearsall Field Austin Chalk'
    if real_data is not None:
        title += ' (Synthetic + Real Data)'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    return fig

def create_data_comparison_table(synthetic_data, real_data, well_name):
    """Create comparison table between synthetic and real data"""
    if real_data is None:
        return None
    
    comparison_data = {
        'Parameter': ['Number of Points', 'Time Range (hours)', 'ΔP Range (psi)', 
                      'Derivative Range (psi)', 'Data Source'],
        'Synthetic': [
            len(synthetic_data[0]),
            f"{synthetic_data[0][0]:.4f} - {synthetic_data[0][-1]:.1f}",
            f"{synthetic_data[1].min():.1f} - {synthetic_data[1].max():.1f}",
            f"{synthetic_data[2].min():.1f} - {synthetic_data[2].max():.1f}",
            'Generated Model'
        ],
        'Real': [
            len(real_data),
            f"{real_data['Time'].min():.4f} - {real_data['Time'].max():.1f}",
            f"{real_data['Delta_P'].min():.1f} - {real_data['Delta_P'].max():.1f}" if 'Delta_P' in real_data.columns else 'N/A',
            f"{real_data['Derivative'].min():.1f} - {real_data['Derivative'].max():.1f}" if 'Derivative' in real_data.columns else 'N/A',
            'Uploaded File'
        ]
    }
    
    return pd.DataFrame(comparison_data)

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ============================================================================
# 3. STREAMLIT APP
# ============================================================================

def main():
    # Title and header
    st.title("📊 Austin Chalk Pressure Transient Analysis")
    st.markdown("**SPE 20609: Pressure Buildup Test Results From Horizontal Wells in the Pearsall Field of the Austin Chalk**")
    
    # Sidebar for controls
    st.sidebar.header("Data Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use Synthetic Data Only", "Upload Real Data + Synthetic"]
    )
    
    # Initialize variables
    real_data = None
    
    # File upload if selected
    if data_source == "Upload Real Data + Synthetic":
        st.sidebar.subheader("Upload Real Data")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv', 'xlsx', 'txt'],
            help="Upload CSV file with columns: 'Time', 'Delta_P', 'Derivative' (optional)"
        )
        
        if uploaded_file is not None:
            try:
                # Try different file formats
                if uploaded_file.name.endswith('.csv'):
                    real_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    real_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    real_data = pd.read_csv(uploaded_file, sep='\t')
                
                st.sidebar.success(f"File uploaded successfully! Shape: {real_data.shape}")
                
                # Show uploaded data preview
                with st.sidebar.expander("Preview Uploaded Data"):
                    st.dataframe(real_data.head())
                
                # Check for required columns
                if 'Time' not in real_data.columns:
                    st.sidebar.error("Uploaded file must contain 'Time' column")
                    real_data = None
                else:
                    # Ensure Time is sorted
                    real_data = real_data.sort_values('Time')
                    
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
    
    # Well selection
    st.sidebar.subheader("Simulation Controls")
    well_options = {
        "Well A - Typical Producer": "typical",
        "Well B - Damaged Well (High Skin)": "damaged", 
        "Well D - Good Producer (Clear Dual Porosity)": "good_producer"
    }
    
    selected_well_name = st.sidebar.selectbox(
        "Select Well Type",
        list(well_options.keys())
    )
    
    well_type = well_options[selected_well_name]
    
    # Plot controls
    st.sidebar.subheader("Plot Options")
    show_log_log = st.sidebar.checkbox("Log-Log Diagnostic Plot", value=True)
    show_sqrt_time = st.sidebar.checkbox("Square Root Time Plot", value=True)
    show_typecurve = st.sidebar.checkbox("Type-Curve Match", value=True)
    show_larsen = st.sidebar.checkbox("Larsen Plot", value=True)
    show_schematic = st.sidebar.checkbox("Schematic Plot", value=True)
    
    # Initialize simulator
    simulator = AustinChalkPressureTransient()
    
    # Generate synthetic data
    time, delta_p, derivative = simulator.generate_very_smooth_data(well_type)
    
    # Main content
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.subheader(f"Analysis for {selected_well_name}")
        if real_data is not None:
            st.success("✅ Real data loaded successfully")
    
    with col2:
        st.metric("Permeability (k)", f"{simulator.k:.1f} md")
        st.metric("Net Thickness (h)", f"{simulator.h:.0f} ft")
    
    with col3:
        st.metric("Storativity Ratio (ω)", f"{simulator.omega:.3f}")
        st.metric("Well Length (Lw)", f"{simulator.Lw:.0f} ft")
    
    # Data comparison if real data is uploaded
    if real_data is not None:
        st.subheader("📋 Data Comparison")
        comparison_df = create_data_comparison_table(
            (time, delta_p, derivative), 
            real_data, 
            selected_well_name
        )
        if comparison_df is not None:
            st.dataframe(comparison_df, use_container_width=True)
    
    # Create synthetic dataframe for download
    synthetic_df = pd.DataFrame({
        'Time (hours)': time,
        'Delta_P (psi)': delta_p,
        'Derivative (psi)': derivative,
        'Data_Source': 'Synthetic'
    })
    
    # Display plots based on selections
    figure_counter = 7
    
    if show_log_log:
        st.subheader(f"Figure {figure_counter}: Log-Log Diagnostic Plot")
        fig = create_diagnostic_log_log_plot(selected_well_name, time, delta_p, derivative, 
                                            figure_counter, real_data)
        st.pyplot(fig)
        figure_counter += 1
    
    if show_sqrt_time:
        st.subheader(f"Figure {figure_counter}: Square Root of Time Plot")
        fig = create_sqrt_time_plot(selected_well_name, time, delta_p, 
                                   figure_counter, real_data)
        st.pyplot(fig)
        figure_counter += 1
    
    if show_typecurve and "Damaged" not in selected_well_name:
        st.subheader(f"Figure {figure_counter}: Type-Curve Match")
        fig = create_type_curve_match(selected_well_name, time, delta_p, derivative, 
                                     figure_counter, real_data)
        st.pyplot(fig)
        figure_counter += 1
    
    if show_larsen and ("Well A" in selected_well_name or "Well D" in selected_well_name):
        st.subheader(f"Figure {figure_counter}: Larsen Plot")
        fig = create_larsen_plot(selected_well_name, time, derivative, 
                                figure_counter, real_data)
        st.pyplot(fig)
        figure_counter += 1
    
    if show_schematic:
        st.subheader("Schematic Flow Model (Figure 5)")
        fig_schematic = create_schematic_plot(simulator, real_data)
        st.pyplot(fig_schematic)
    
    # Results summary table
    st.subheader("📊 Summary of Results (Simulating Table 3 from Paper)")
    
    results_data = {
        'Well': ['A', 'B', 'C', 'D'],
        'k_f (md)': [5.2, 0.8, 3.5, 7.8],
        'h (ft)': [180, 120, 250, 320],
        'ω': [0.03, 'N/A', 0.05, 0.02],
        'λ': ['1.2E-07', 'N/A', '8.5E-08', '5.6E-08'],
        'C (bbl/psi)': [85, 45, 120, 95],
        'Skin (S_m)': [-1.5, 13.2, -2.1, -2.8],
        'Flow Regime': ['Linear + DP', 'Radial + Linear', 'Linear', 'Linear + Clear DP']
    }
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Data download section
    st.subheader("📥 Data Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Synthetic data download
        st.markdown("### Synthetic Data")
        csv_synthetic = synthetic_df.to_csv(index=False)
        st.download_button(
            label="Download Synthetic Data (CSV)",
            data=csv_synthetic,
            file_name=f"synthetic_{selected_well_name.replace(' ', '_')}_pressure_data.csv",
            mime="text/csv"
        )
        
        # Display sample of synthetic data
        with st.expander("Preview Synthetic Data"):
            st.dataframe(synthetic_df.head(10))
    
    with col2:
        # Combined data download (if real data exists)
        if real_data is not None:
            st.markdown("### Combined Data")
            
            # Prepare combined dataframe
            combined_df = synthetic_df.copy()
            combined_df['Data_Source'] = 'Synthetic'
            
            real_df_formatted = real_data.copy()
            # Ensure consistent column names
            if 'Delta_P' not in real_df_formatted.columns and 'Pressure' in real_df_formatted.columns:
                real_df_formatted = real_df_formatted.rename(columns={'Pressure': 'Delta_P'})
            if 'Derivative' not in real_df_formatted.columns:
                real_df_formatted['Derivative'] = np.nan
            
            real_df_formatted['Data_Source'] = 'Real'
            real_df_formatted = real_df_formatted.rename(columns={
                'Time': 'Time (hours)',
                'Delta_P': 'Delta_P (psi)',
                'Derivative': 'Derivative (psi)'
            })
            
            # Combine data
            combined_data = pd.concat([combined_df, real_df_formatted], ignore_index=True)
            
            csv_combined = combined_data.to_csv(index=False)
            st.download_button(
                label="Download Combined Data (CSV)",
                data=csv_combined,
                file_name=f"combined_{selected_well_name.replace(' ', '_')}_pressure_data.csv",
                mime="text/csv"
            )
            
            # Display sample of combined data
            with st.expander("Preview Combined Data"):
                st.dataframe(combined_data.head(10))
    
    # Key findings
    st.subheader("🔑 Key Findings from the Analysis")
    findings = [
        "1. Predominant linear flow in most wells",
        "2. Dual porosity transitions with ω between 0.01-0.1",
        "3. Fracture permeabilities in 0.5-10 md range",
        "4. Effective drained thickness: 100-400 ft",
        "5. Wellbore storage effects are significant in early time",
        "6. Damaged wells show high skin factor and radial flow"
    ]
    
    for finding in findings:
        st.markdown(f"• {finding}")
    
    # File format guidance
    st.subheader("📄 File Format Guidance")
    st.markdown("""
    **For uploading real data, your CSV file should have the following columns:**
    - `Time` - Elapsed time in hours
    - `Delta_P` - Pressure change in psi (required)
    - `Derivative` - Pressure derivative in psi (optional)
    
    **Example CSV format:**
    ```
    Time,Delta_P,Derivative
    0.001,5.2,10.5
    0.01,15.8,12.3
    0.1,28.9,14.7
    1.0,45.6,16.2
    10.0,68.3,18.9
    ```
    """)

if __name__ == "__main__":
    main()

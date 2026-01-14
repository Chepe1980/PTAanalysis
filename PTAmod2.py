import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.interpolate import CubicSpline
import io
import base64

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
# 2. PLOTLY PLOTTING FUNCTIONS (FIXED FOR PANDAS SERIES)
# ============================================================================

def create_diagnostic_log_log_plot_plotly(well_name, time, delta_p, derivative, fig_num, data_source="Synthetic"):
    """Create interactive log-log diagnostic plot using Plotly"""
    # Convert to numpy arrays to avoid pandas indexing issues
    time_np = np.array(time) if not isinstance(time, np.ndarray) else time
    delta_p_np = np.array(delta_p) if not isinstance(delta_p, np.ndarray) else delta_p
    derivative_np = np.array(derivative) if not isinstance(derivative, np.ndarray) else derivative
    
    fig = go.Figure()
    
    # Add pressure change trace
    fig.add_trace(go.Scatter(
        x=time_np,
        y=delta_p_np,
        mode='lines+markers',
        name=f'{data_source} ΔP (psi)',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        opacity=0.7
    ))
    
    # Add derivative trace
    fig.add_trace(go.Scatter(
        x=time_np,
        y=derivative_np,
        mode='lines+markers',
        name=f'{data_source} Derivative (psi)',
        line=dict(color='red', width=2),
        marker=dict(size=4, symbol='square'),
        opacity=0.7
    ))
    
    # Add reference lines for different flow regimes
    # Unit slope for wellbore storage
    unit_slope_x = np.array([time_np[0], time_np[-1]])
    unit_slope_y = 10 * (unit_slope_x / unit_slope_x[0])
    fig.add_trace(go.Scatter(
        x=unit_slope_x,
        y=unit_slope_y,
        mode='lines',
        name='Unit Slope (Wellbore Storage)',
        line=dict(color='black', dash='dash', width=1),
        opacity=0.5
    ))
    
    # Half slope for linear flow
    half_slope_x = np.array([1e-1, 1e2])
    half_slope_y = 100 * np.sqrt(half_slope_x / half_slope_x[0])
    fig.add_trace(go.Scatter(
        x=half_slope_x,
        y=half_slope_y,
        mode='lines',
        name='½ Slope (Linear Flow)',
        line=dict(color='green', dash='dash', width=1),
        opacity=0.5
    ))
    
    # Constant derivative for radial flow
    mask_radial = (time_np > 0.01) & (time_np < 0.1)
    if np.sum(mask_radial) > 0:
        radial_level = np.mean(derivative_np[mask_radial])
        if radial_level > 0:
            fig.add_trace(go.Scatter(
                x=[time_np[0], time_np[-1]],
                y=[radial_level, radial_level],
                mode='lines',
                name=f'Radial Flow Level ({radial_level:.1f} psi)',
                line=dict(color='magenta', dash='dash', width=1),
                opacity=0.5
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Figure {fig_num}: Log-Log Diagnostic Plot - {well_name}',
            font=dict(size=18, family="Arial, sans-serif", color='darkblue'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Elapsed Time, Δt (hours)',
            type='log',
            title_font=dict(size=14, family="Arial, sans-serif"),
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title='Pressure Change and Derivative (psi)',
            type='log',
            title_font=dict(size=14, family="Arial, sans-serif"),
            gridcolor='lightgray',
            gridwidth=1
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        template='plotly_white',
        hovermode='x unified',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add annotations for flow regimes
    fig.add_annotation(
        x=0.005,
        y=1,
        text="Wellbore<br>Storage",
        showarrow=False,
        font=dict(size=10, color='black'),
        bgcolor='rgba(255, 255, 0, 0.3)',
        bordercolor='black',
        borderwidth=1,
        borderpad=4
    )
    
    if 'damaged' not in well_name.lower():
        fig.add_annotation(
            x=0.5,
            y=10,
            text="Linear Flow",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='rgba(144, 238, 144, 0.3)',
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            textangle=26
        )
        
        # Mark dual porosity transition if visible
        mask_transition = (time_np > 0.1) & (time_np < 10)
        if np.sum(mask_transition) > 0:
            max_deriv_in_range = np.max(derivative_np[mask_transition])
            if np.min(derivative_np) < 0.5 * max_deriv_in_range:
                fig.add_annotation(
                    x=5,
                    y=np.min(derivative_np)*0.8,
                    text="Dual Porosity<br>Transition",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255, 165, 0, 0.3)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=4
                )
    
    return fig

def create_sqrt_time_plot_plotly(well_name, time, delta_p, fig_num, data_source="Synthetic"):
    """Create interactive square root of time plot using Plotly"""
    # Convert to numpy arrays
    time_np = np.array(time) if not isinstance(time, np.ndarray) else time
    delta_p_np = np.array(delta_p) if not isinstance(delta_p, np.ndarray) else delta_p
    
    fig = go.Figure()
    
    # Calculate sqrt(time)
    sqrt_time = np.sqrt(time_np)
    
    # Plot pressure vs sqrt(time)
    fig.add_trace(go.Scatter(
        x=sqrt_time,
        y=delta_p_np,
        mode='lines+markers',
        name=f'{data_source} Data',
        line=dict(color='blue', width=2),
        marker=dict(size=5),
        opacity=0.7
    ))
    
    # Identify linear flow segments
    # Early linear flow (pre-transition)
    mask_early = (time_np > 0.1) & (time_np < 1.0)
    if np.sum(mask_early) > 5:
        coeff_early = np.polyfit(sqrt_time[mask_early], delta_p_np[mask_early], 1)
        line_early = np.polyval(coeff_early, sqrt_time[mask_early])
        m1 = coeff_early[0]
        
        fig.add_trace(go.Scatter(
            x=sqrt_time[mask_early],
            y=line_early,
            mode='lines',
            name=f'Early Linear Flow (m₁ = {m1:.1f})',
            line=dict(color='red', width=3),
            opacity=0.8
        ))
        
        # Add annotation for m1
        fig.add_annotation(
            x=np.mean(sqrt_time[mask_early]),
            y=np.mean(delta_p_np[mask_early]),
            text=f'm₁ = {m1:.1f}',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red',
            font=dict(size=11, color='red'),
            bgcolor='rgba(255, 0, 0, 0.2)',
            bordercolor='red',
            borderwidth=1
        )
    
    # Late linear flow (post-transition)
    mask_late = (time_np > 10) & (time_np < 100)
    if np.sum(mask_late) > 5:
        coeff_late = np.polyfit(sqrt_time[mask_late], delta_p_np[mask_late], 1)
        line_late = np.polyval(coeff_late, sqrt_time[mask_late])
        m2 = coeff_late[0]
        
        fig.add_trace(go.Scatter(
            x=sqrt_time[mask_late],
            y=line_late,
            mode='lines',
            name=f'Late Linear Flow (m₂ = {m2:.1f})',
            line=dict(color='green', width=3),
            opacity=0.8
        ))
        
        # Add annotation for m2
        fig.add_annotation(
            x=np.mean(sqrt_time[mask_late]),
            y=np.mean(delta_p_np[mask_late]),
            text=f'm₂ = {m2:.1f}',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='green',
            font=dict(size=11, color='green'),
            bgcolor='rgba(0, 255, 0, 0.2)',
            bordercolor='green',
            borderwidth=1
        )
        
        # Calculate storativity ratio ω = m1/m2
        if 'm1' in locals():
            omega_calc = (m1/m2)**2
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'ω ≈ {omega_calc:.3f}',
                showarrow=False,
                font=dict(size=14, color='black'),
                bgcolor='rgba(255, 255, 0, 0.5)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Figure {fig_num}: Square Root of Time Plot - {well_name}',
            font=dict(size=18, family="Arial, sans-serif", color='darkblue'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='√(Equivalent Time) (√hours)',
            title_font=dict(size=14, family="Arial, sans-serif"),
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title='Pressure Change, ΔP (psi)',
            title_font=dict(size=14, family="Arial, sans-serif"),
            gridcolor='lightgray',
            gridwidth=1
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        template='plotly_white',
        hovermode='x unified',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_larsen_plot_plotly(well_name, time, derivative, fig_num, data_source="Synthetic"):
    """Create interactive Larsen plot using Plotly"""
    # Convert to numpy arrays
    time_np = np.array(time) if not isinstance(time, np.ndarray) else time
    derivative_np = np.array(derivative) if not isinstance(derivative, np.ndarray) else derivative
    
    fig = go.Figure()
    
    # Calculate sqrt(time)
    sqrt_time = np.sqrt(time_np)
    
    # Filter out very early times
    mask = time_np > 0.001
    sqrt_time_filt = sqrt_time[mask]
    deriv_filt = derivative_np[mask]
    
    # Plot derivative vs sqrt(time)
    fig.add_trace(go.Scatter(
        x=sqrt_time_filt,
        y=deriv_filt,
        mode='lines+markers',
        name=f'{data_source} Data',
        line=dict(color='blue', width=2),
        marker=dict(size=5),
        opacity=0.7
    ))
    
    # Identify linear flow region
    mask_linear = (time_np > 0.1) & (time_np < 10)
    if np.sum(mask_linear) > 5:
        sqrt_linear = sqrt_time[mask_linear]
        deriv_linear = derivative_np[mask_linear]
        
        # Fit line through origin
        slope = np.sum(sqrt_linear * deriv_linear) / np.sum(sqrt_linear**2)
        
        # Plot fitted line
        x_fit = np.linspace(0, np.max(sqrt_linear), 50)
        y_fit = slope * x_fit
        
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f'Linear Flow Fit: slope = {slope:.2f}',
            line=dict(color='red', width=3, dash='dash'),
            opacity=0.8
        ))
        
        # Mark upper limit for radial flow (m*)
        if len(deriv_linear) > 0:
            m_star = deriv_linear[0]
            fig.add_trace(go.Scatter(
                x=[0, np.max(sqrt_time_filt)],
                y=[m_star, m_star],
                mode='lines',
                name=f'm* = {m_star:.1f} psi (Radial Flow Limit)',
                line=dict(color='green', width=2, dash='dash'),
                opacity=0.7
            ))
            
            # Mark radial flow region
            mask_radial = time_np < 0.1
            if np.sum(mask_radial) > 3:
                fig.add_trace(go.Scatter(
                    x=sqrt_time[mask_radial],
                    y=derivative_np[mask_radial],
                    mode='markers',
                    name='Radial Flow Region',
                    marker=dict(size=8, color='magenta', symbol='square'),
                    opacity=0.8
                ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Figure {fig_num}: Larsen Plot - {well_name}',
            font=dict(size=18, family="Arial, sans-serif", color='darkblue'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='√(Elapsed Time) (√hours)',
            title_font=dict(size=14, family="Arial, sans-serif"),
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title='Pressure Derivative (psi)',
            title_font=dict(size=14, family="Arial, sans-serif"),
            gridcolor='lightgray',
            gridwidth=1
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        template='plotly_white',
        hovermode='x unified',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add explanatory text
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref='paper',
        yref='paper',
        text='Larsen Plot Analysis:',
        showarrow=False,
        font=dict(size=12, color='darkblue'),
        align='left'
    )
    
    fig.add_annotation(
        x=0.05,
        y=0.90,
        xref='paper',
        yref='paper',
        text='1. Radial flow → Constant derivative',
        showarrow=False,
        font=dict(size=11, color='black'),
        align='left'
    )
    
    fig.add_annotation(
        x=0.05,
        y=0.85,
        xref='paper',
        yref='paper',
        text='2. Linear flow → Line through origin',
        showarrow=False,
        font=dict(size=11, color='black'),
        align='left'
    )
    
    fig.add_annotation(
        x=0.05,
        y=0.80,
        xref='paper',
        yref='paper',
        text='3. m* gives lower limit for k',
        showarrow=False,
        font=dict(size=11, color='black'),
        align='left'
    )
    
    return fig

def create_type_curve_match_plotly(well_name, time, delta_p, derivative, fig_num, data_source="Synthetic"):
    """Create interactive type-curve match plot using Plotly"""
    # Convert to numpy arrays
    time_np = np.array(time) if not isinstance(time, np.ndarray) else time
    delta_p_np = np.array(delta_p) if not isinstance(delta_p, np.ndarray) else delta_p
    derivative_np = np.array(derivative) if not isinstance(derivative, np.ndarray) else derivative
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Dimensionless Pressure', 'Dimensionless Derivative'),
        vertical_spacing=0.1
    )
    
    # Generate theoretical type curves for different ω values
    tD_range = np.logspace(-2, 4, 100)
    
    # Cinco-Ley & Meng type curves for linear flow in dual porosity
    omega_values = [0.001, 0.01, 0.05, 0.1]
    colors = ['red', 'green', 'blue', 'magenta']
    
    for omega, color in zip(omega_values, colors):
        # Simplified dual porosity response for linear flow
        pD_early = 2 * np.sqrt(tD_range / (np.pi * omega))
        transition_point = -np.log(omega) / 10
        pD_late = 2 * np.sqrt(tD_range / np.pi)
        
        # Combine with smooth transition
        weight = 1 / (1 + np.exp(-10*(np.log10(tD_range) - np.log10(transition_point))))
        pD = (1 - weight) * pD_early + weight * pD_late
        
        # Derivative
        deriv_D = 0.5 * pD
        
        # Apply dual porosity "dip"
        if omega < 0.1:
            dip_center = transition_point
            dip_width = 1.0
            dip_strength = 0.5 * (0.1 - omega) / 0.1
            dip_factor = 1 - dip_strength * np.exp(-((np.log10(tD_range) - np.log10(dip_center))**2) / (2*dip_width**2))
            deriv_D *= dip_factor
        
        # Add type curves to subplots
        fig.add_trace(go.Scatter(
            x=tD_range,
            y=pD,
            mode='lines',
            name=f'ω = {omega}',
            line=dict(color=color, width=1.5),
            opacity=0.7,
            showlegend=True,
            legendgroup=f'omega_{omega}'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=tD_range,
            y=deriv_D,
            mode='lines',
            name=f'ω = {omega}',
            line=dict(color=color, width=1.5),
            opacity=0.7,
            showlegend=False,
            legendgroup=f'omega_{omega}'
        ), row=2, col=1)
    
    # Scale field data to match type curves
    scale_time = 10.0
    scale_pressure = 100.0
    
    tD_field = time_np * scale_time
    pD_field = delta_p_np / scale_pressure
    # Use numpy gradient with log of time for better derivative calculation
    derivD_field = np.gradient(pD_field, np.log(tD_field))
    
    # Add data to subplots
    fig.add_trace(go.Scatter(
        x=tD_field,
        y=pD_field,
        mode='lines+markers',
        name=f'{data_source} Data',
        line=dict(color='black', width=2),
        marker=dict(size=5),
        opacity=0.8,
        showlegend=True,
        legendgroup='data'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=tD_field,
        y=derivD_field,
        mode='lines+markers',
        name=f'{data_source} Data',
        line=dict(color='black', width=2),
        marker=dict(size=5),
        opacity=0.8,
        showlegend=False,
        legendgroup='data'
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Figure {fig_num}: Cinco-Ley & Meng Type-Curve Match - {well_name}',
            font=dict(size=18, family="Arial, sans-serif", color='darkblue'),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_white',
        height=800,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        title_text='Dimensionless Time, tD',
        type='log',
        row=2, col=1
    )
    
    fig.update_xaxes(
        type='log',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Dimensionless Pressure, pD',
        type='log',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Dimensionless Derivative',
        type='log',
        row=2, col=1
    )
    
    # Add match information
    match_text = f'Time Match: tD/t = {scale_time:.1f}<br>Pressure Match: pD/ΔP = {1/scale_pressure:.4f}'
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref='paper',
        yref='paper',
        text=match_text,
        showarrow=False,
        font=dict(size=11, color='black'),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1,
        borderpad=4,
        align='left'
    )
    
    # Indicate estimated ω
    if 'good_producer' in well_name.lower() or 'd' in well_name.lower():
        fig.add_annotation(
            x=0.05,
            y=0.85,
            xref='paper',
            yref='paper',
            text='Best Match: ω ≈ 0.01-0.05',
            showarrow=False,
            font=dict(size=11, color='black'),
            bgcolor='rgba(255, 255, 0, 0.5)',
            bordercolor='black',
            borderwidth=1,
            borderpad=4
        )
    
    return fig

def create_data_summary_table(data, data_source):
    """Create an interactive data summary table"""
    if data is None:
        return None
    
    summary_data = {
        'Parameter': ['Data Points', 'Time Range (hours)', 'Min ΔP (psi)', 'Max ΔP (psi)', 
                      'Min Derivative (psi)', 'Max Derivative (psi)', 'Data Source'],
        'Value': [
            len(data),
            f"{data['Time'].min():.4f} - {data['Time'].max():.1f}",
            f"{data['Delta_P'].min():.2f}" if 'Delta_P' in data.columns else 'N/A',
            f"{data['Delta_P'].max():.2f}" if 'Delta_P' in data.columns else 'N/A',
            f"{data['Derivative'].min():.2f}" if 'Derivative' in data.columns else 'N/A',
            f"{data['Derivative'].max():.2f}" if 'Derivative' in data.columns else 'N/A',
            data_source
        ]
    }
    
    return pd.DataFrame(summary_data)

# ============================================================================
# 3. STREAMLIT APP (FIXED VERSION)
# ============================================================================

def main():
    # Title and header
    st.title("📊 Austin Chalk Pressure Transient Analysis")
    st.markdown("**SPE 20609: Pressure Buildup Test Results From Horizontal Wells in the Pearsall Field of the Austin Chalk**")
    
    # Initialize session state for data persistence
    if 'real_data' not in st.session_state:
        st.session_state.real_data = None
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "Synthetic Only"
    
    # Sidebar for controls
    st.sidebar.header("Data Configuration")
    
    # Data source selection
    st.session_state.data_source = st.sidebar.radio(
        "Select Data Source",
        ["Synthetic Only", "Real Data Only", "Both Synthetic and Real"]
    )
    
    # File upload for real data
    real_data = None
    if st.session_state.data_source in ["Real Data Only", "Both Synthetic and Real"]:
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
                
                st.sidebar.success(f"File uploaded successfully! {len(real_data)} data points")
                
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
                    # Store in session state
                    st.session_state.real_data = real_data
                    
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
        elif st.session_state.real_data is not None:
            # Use previously uploaded data
            real_data = st.session_state.real_data
    
    # Well selection for synthetic data
    synthetic_df = None
    selected_well_name = ""
    simulator = None
    
    if st.session_state.data_source in ["Synthetic Only", "Both Synthetic and Real"]:
        st.sidebar.subheader("Synthetic Data Configuration")
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
        
        # Initialize simulator
        simulator = AustinChalkPressureTransient()
        
        # Generate synthetic data
        time, delta_p, derivative = simulator.generate_very_smooth_data(well_type)
        
        # Create synthetic dataframe
        synthetic_df = pd.DataFrame({
            'Time': time,
            'Delta_P': delta_p,
            'Derivative': derivative
        })
        
        # Store in session state
        st.session_state.synthetic_data = {
            'df': synthetic_df,
            'well_name': selected_well_name,
            'simulator': simulator
        }
    elif st.session_state.synthetic_data is not None:
        # Use previously generated synthetic data
        synthetic_data = st.session_state.synthetic_data
        synthetic_df = synthetic_data['df']
        selected_well_name = synthetic_data['well_name']
        simulator = synthetic_data['simulator']
    
    # Plot controls
    st.sidebar.subheader("Plot Options")
    show_log_log = st.sidebar.checkbox("Log-Log Diagnostic Plot", value=True)
    show_sqrt_time = st.sidebar.checkbox("Square Root Time Plot", value=True)
    show_typecurve = st.sidebar.checkbox("Type-Curve Match", value=True)
    show_larsen = st.sidebar.checkbox("Larsen Plot", value=True)
    show_schematic = st.sidebar.checkbox("Schematic Plot", value=True)
    
    # Main content
    st.header("📈 Pressure Transient Analysis")
    
    # Display data source info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data_source == "Synthetic Only":
            st.info("📊 Using Synthetic Data Only")
        elif st.session_state.data_source == "Real Data Only":
            st.info("📊 Using Real Data Only")
        else:
            st.info("📊 Using Both Synthetic and Real Data")
    
    with col2:
        if st.session_state.data_source in ["Synthetic Only", "Both Synthetic and Real"] and synthetic_df is not None:
            st.metric("Synthetic Data Points", len(synthetic_df))
    
    with col3:
        if real_data is not None:
            st.metric("Real Data Points", len(real_data))
    
    # Data summary table
    st.subheader("📋 Data Summary")
    
    if st.session_state.data_source == "Synthetic Only" and synthetic_df is not None:
        summary_df = create_data_summary_table(synthetic_df, "Synthetic")
        if summary_df is not None:
            st.dataframe(summary_df, use_container_width=True)
    
    elif st.session_state.data_source == "Real Data Only" and real_data is not None:
        summary_df = create_data_summary_table(real_data, "Real")
        if summary_df is not None:
            st.dataframe(summary_df, use_container_width=True)
    
    elif st.session_state.data_source == "Both Synthetic and Real":
        col1, col2 = st.columns(2)
        
        with col1:
            if synthetic_df is not None:
                summary_synth = create_data_summary_table(synthetic_df, "Synthetic")
                if summary_synth is not None:
                    st.dataframe(summary_synth, use_container_width=True)
        
        with col2:
            if real_data is not None:
                summary_real = create_data_summary_table(real_data, "Real")
                if summary_real is not None:
                    st.dataframe(summary_real, use_container_width=True)
    
    # Create and display plots
    figure_counter = 7
    
    if show_log_log:
        st.subheader(f"Figure {figure_counter}: Log-Log Diagnostic Plot")
        
        if st.session_state.data_source == "Synthetic Only" and synthetic_df is not None:
            fig = create_diagnostic_log_log_plot_plotly(
                selected_well_name, 
                synthetic_df['Time'], 
                synthetic_df['Delta_P'], 
                synthetic_df['Derivative'], 
                figure_counter, 
                "Synthetic"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.data_source == "Real Data Only" and real_data is not None:
            if 'Delta_P' in real_data.columns:
                # Calculate derivative if not present
                if 'Derivative' not in real_data.columns:
                    # Simple numerical derivative
                    real_data['Derivative'] = np.gradient(
                        real_data['Delta_P'].values, 
                        np.log(real_data['Time'].values)
                    )
                
                fig = create_diagnostic_log_log_plot_plotly(
                    "Real Well Data", 
                    real_data['Time'], 
                    real_data['Delta_P'], 
                    real_data['Derivative'], 
                    figure_counter, 
                    "Real"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.data_source == "Both Synthetic and Real":
            # Create two separate plots side by side
            col1, col2 = st.columns(2)
            
            with col1:
                if synthetic_df is not None:
                    fig = create_diagnostic_log_log_plot_plotly(
                        selected_well_name, 
                        synthetic_df['Time'], 
                        synthetic_df['Delta_P'], 
                        synthetic_df['Derivative'], 
                        figure_counter, 
                        "Synthetic"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if real_data is not None and 'Delta_P' in real_data.columns:
                    if 'Derivative' not in real_data.columns:
                        real_data['Derivative'] = np.gradient(
                            real_data['Delta_P'].values, 
                            np.log(real_data['Time'].values)
                        )
                    
                    fig = create_diagnostic_log_log_plot_plotly(
                        "Real Well Data", 
                        real_data['Time'], 
                        real_data['Delta_P'], 
                        real_data['Derivative'], 
                        figure_counter, 
                        "Real"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        figure_counter += 1
    
    if show_sqrt_time:
        st.subheader(f"Figure {figure_counter}: Square Root of Time Plot")
        
        if st.session_state.data_source == "Synthetic Only" and synthetic_df is not None:
            fig = create_sqrt_time_plot_plotly(
                selected_well_name, 
                synthetic_df['Time'], 
                synthetic_df['Delta_P'], 
                figure_counter, 
                "Synthetic"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.data_source == "Real Data Only" and real_data is not None:
            if 'Delta_P' in real_data.columns:
                fig = create_sqrt_time_plot_plotly(
                    "Real Well Data", 
                    real_data['Time'], 
                    real_data['Delta_P'], 
                    figure_counter, 
                    "Real"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.data_source == "Both Synthetic and Real":
            col1, col2 = st.columns(2)
            
            with col1:
                if synthetic_df is not None:
                    fig = create_sqrt_time_plot_plotly(
                        selected_well_name, 
                        synthetic_df['Time'], 
                        synthetic_df['Delta_P'], 
                        figure_counter, 
                        "Synthetic"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if real_data is not None and 'Delta_P' in real_data.columns:
                    fig = create_sqrt_time_plot_plotly(
                        "Real Well Data", 
                        real_data['Time'], 
                        real_data['Delta_P'], 
                        figure_counter, 
                        "Real"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        figure_counter += 1
    
    if show_typecurve:
        st.subheader(f"Figure {figure_counter}: Type-Curve Match")
        
        well_name_to_check = selected_well_name
        
        if "Damaged" not in well_name_to_check:
            if st.session_state.data_source == "Synthetic Only" and synthetic_df is not None:
                fig = create_type_curve_match_plotly(
                    selected_well_name, 
                    synthetic_df['Time'], 
                    synthetic_df['Delta_P'], 
                    synthetic_df['Derivative'], 
                    figure_counter, 
                    "Synthetic"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.data_source == "Real Data Only" and real_data is not None:
                if 'Delta_P' in real_data.columns:
                    if 'Derivative' not in real_data.columns:
                        real_data['Derivative'] = np.gradient(
                            real_data['Delta_P'].values, 
                            np.log(real_data['Time'].values)
                        )
                    
                    fig = create_type_curve_match_plotly(
                        "Real Well Data", 
                        real_data['Time'], 
                        real_data['Delta_P'], 
                        real_data['Derivative'], 
                        figure_counter, 
                        "Real"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.data_source == "Both Synthetic and Real":
                col1, col2 = st.columns(2)
                
                with col1:
                    if synthetic_df is not None and "Damaged" not in selected_well_name:
                        fig = create_type_curve_match_plotly(
                            selected_well_name, 
                            synthetic_df['Time'], 
                            synthetic_df['Delta_P'], 
                            synthetic_df['Derivative'], 
                            figure_counter, 
                            "Synthetic"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if real_data is not None and 'Delta_P' in real_data.columns:
                        if 'Derivative' not in real_data.columns:
                            real_data['Derivative'] = np.gradient(
                                real_data['Delta_P'].values, 
                                np.log(real_data['Time'].values)
                            )
                        
                        fig = create_type_curve_match_plotly(
                            "Real Well Data", 
                            real_data['Time'], 
                            real_data['Delta_P'], 
                            real_data['Derivative'], 
                            figure_counter, 
                            "Real"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        figure_counter += 1
    
    if show_larsen:
        st.subheader(f"Figure {figure_counter}: Larsen Plot")
        
        if st.session_state.data_source == "Synthetic Only" and synthetic_df is not None:
            if "Well A" in selected_well_name or "Well D" in selected_well_name:
                fig = create_larsen_plot_plotly(
                    selected_well_name, 
                    synthetic_df['Time'], 
                    synthetic_df['Derivative'], 
                    figure_counter, 
                    "Synthetic"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.data_source == "Real Data Only" and real_data is not None:
            if 'Delta_P' in real_data.columns:
                if 'Derivative' not in real_data.columns:
                    real_data['Derivative'] = np.gradient(
                        real_data['Delta_P'].values, 
                        np.log(real_data['Time'].values)
                    )
                
                fig = create_larsen_plot_plotly(
                    "Real Well Data", 
                    real_data['Time'], 
                    real_data['Derivative'], 
                    figure_counter, 
                    "Real"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.data_source == "Both Synthetic and Real":
            col1, col2 = st.columns(2)
            
            with col1:
                if synthetic_df is not None and ("Well A" in selected_well_name or "Well D" in selected_well_name):
                    fig = create_larsen_plot_plotly(
                        selected_well_name, 
                        synthetic_df['Time'], 
                        synthetic_df['Derivative'], 
                        figure_counter, 
                        "Synthetic"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if real_data is not None and 'Delta_P' in real_data.columns:
                    if 'Derivative' not in real_data.columns:
                        real_data['Derivative'] = np.gradient(
                            real_data['Delta_P'].values, 
                            np.log(real_data['Time'].values)
                        )
                    
                    fig = create_larsen_plot_plotly(
                        "Real Well Data", 
                        real_data['Time'], 
                        real_data['Derivative'], 
                        figure_counter, 
                        "Real"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        figure_counter += 1
    
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
    
    if st.session_state.data_source == "Synthetic Only" and synthetic_df is not None:
        csv_synthetic = synthetic_df.to_csv(index=False)
        
        st.download_button(
            label="Download Synthetic Data (CSV)",
            data=csv_synthetic,
            file_name=f"synthetic_{selected_well_name.replace(' ', '_')}_pressure_data.csv",
            mime="text/csv"
        )
    
    elif st.session_state.data_source == "Real Data Only" and real_data is not None:
        csv_real = real_data.to_csv(index=False)
        
        st.download_button(
            label="Download Real Data (CSV)",
            data=csv_real,
            file_name="real_pressure_data.csv",
            mime="text/csv"
        )
    
    elif st.session_state.data_source == "Both Synthetic and Real":
        col1, col2 = st.columns(2)
        
        with col1:
            if synthetic_df is not None:
                csv_synthetic = synthetic_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Synthetic Data (CSV)",
                    data=csv_synthetic,
                    file_name=f"synthetic_{selected_well_name.replace(' ', '_')}_pressure_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if real_data is not None:
                csv_real = real_data.to_csv(index=False)
                
                st.download_button(
                    label="Download Real Data (CSV)",
                    data=csv_real,
                    file_name="real_pressure_data.csv",
                    mime="text/csv"
                )
    
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
    - `Derivative` - Pressure derivative in psi (optional - will be calculated if not provided)
    
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

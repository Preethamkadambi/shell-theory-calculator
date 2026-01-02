import streamlit as st
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="Shell Theory Calculator", layout="wide")

st.title("StructAeroShells: Shell & Plate Theory Calculator")
st.markdown("""
This tool calculates Stress Resultants (Forces & Moments) based on the 
**Constitutive Laws** defined in the `StructAeroShells.pdf`.
""")

# --- Sidebar Inputs ---
st.sidebar.header("1. Material & Geometry")
E = st.sidebar.number_input("Young's Modulus (E) [Pa]", value=70e9, format="%.2e")
nu = st.sidebar.number_input("Poisson's Ratio (ν)", value=0.3)
h = st.sidebar.number_input("Thickness (h) [m]", value=0.01, format="%.4f")

# Derived constants
C = E / (1 - nu**2)
G = E / (2 * (1 + nu))

st.sidebar.markdown("---")
st.sidebar.header("2. Input Strains (Generalized)")

# Membrane Strains
st.sidebar.subheader("Membrane Strains (ε)")
eps_11 = st.sidebar.number_input("ε_11", value=0.001, format="%.6f")
eps_22 = st.sidebar.number_input("ε_22", value=0.0)
gam_12 = st.sidebar.number_input("γ_12 (Engineering)", value=0.0)
eps = np.array([eps_11, eps_22, gam_12])

# Bending Curvatures
st.sidebar.subheader("Curvatures (κ)")
kap_11 = st.sidebar.number_input("κ_11 [1/m]", value=0.0)
kap_22 = st.sidebar.number_input("κ_22 [1/m]", value=0.5)
kap_12 = st.sidebar.number_input("κ_12 (Twist) [1/m]", value=0.0)
kap = np.array([kap_11, kap_22, kap_12])

# Transverse Shear
st.sidebar.subheader("Transverse Shear (γ)")
gam_13 = st.sidebar.number_input("γ_13", value=0.0)
gam_23 = st.sidebar.number_input("γ_23", value=0.0)
gam_z = np.array([gam_13, gam_23])

# --- Tabs for Cases ---
tab1, tab2, tab3 = st.tabs(["1. Reissner-Mindlin", "2. Kirchhoff-Love", "3. Linear Shells"])

def format_matrix(mat):
    return pd.DataFrame(mat, columns=['11', '22', '12'], index=['11', '22', '12'])

# --- Common Matrices Calculation ---
# Base Matrix (Voigt notation for isotropic material)
mat_base = np.array([
    [1,      nu,     0],
    [nu,     1,      0],
    [0,      0,      (1-nu)/2] 
])

H_n = C * h * mat_base
H_m = C * (h**3 / 12.0) * mat_base

# --- TAB 1: Reissner-Mindlin ---
with tab1:
    st.header("Reissner-Mindlin Plate (Thick)")
    st.markdown("Reference: **Pages 3-10**")
    
    k_shear = st.number_input("Shear Correction Factor (k)", value=0.8333, key="k_rm")
    H_q = k_shear * G * h * np.eye(2)

    if st.button("Calculate RM Resultants"):
        N = H_n @ eps
        M = H_m @ kap
        Q = H_q @ gam_z
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Membrane Forces (N)")
            st.latex(r"\begin{bmatrix} N_{11} \\ N_{22} \\ N_{12} \end{bmatrix} = " + 
                     r"\begin{bmatrix} " + f"{N[0]:.4e} \\\\ {N[1]:.4e} \\\\ {N[2]:.4e}" + r" \end{bmatrix}")
        with col2:
            st.subheader("Bending Moments (M)")
            st.latex(r"\begin{bmatrix} M_{11} \\ M_{22} \\ M_{12} \end{bmatrix} = " + 
                     r"\begin{bmatrix} " + f"{M[0]:.4e} \\\\ {M[1]:.4e} \\\\ {M[2]:.4e}" + r" \end{bmatrix}")
        with col3:
            st.subheader("Shear Forces (Q)")
            st.latex(r"\begin{bmatrix} Q_{1} \\ Q_{2} \end{bmatrix} = " + 
                     r"\begin{bmatrix} " + f"{Q[0]:.4e} \\\\ {Q[1]:.4e}" + r" \end{bmatrix}")

# --- TAB 2: Kirchhoff-Love ---
with tab2:
    st.header("Kirchhoff-Love Plate (Thin)")
    st.markdown("Reference: **Pages 11-12**")
    st.info("Assumption: Transverse shear strains are zero/negligible. Shear forces are reactive.")

    if st.button("Calculate KL Resultants"):
        N = H_n @ eps
        M = H_m @ kap
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Membrane Forces (N)")
            st.latex(r"\begin{bmatrix} N_{11} \\ N_{22} \\ N_{12} \end{bmatrix} = " + 
                     r"\begin{bmatrix} " + f"{N[0]:.4e} \\\\ {N[1]:.4e} \\\\ {N[2]:.4e}" + r" \end{bmatrix}")
        with col2:
            st.subheader("Bending Moments (M)")
            st.latex(r"\begin{bmatrix} M_{11} \\ M_{22} \\ M_{12} \end{bmatrix} = " + 
                     r"\begin{bmatrix} " + f"{M[0]:.4e} \\\\ {M[1]:.4e} \\\\ {M[2]:.4e}" + r" \end{bmatrix}")

# --- TAB 3: Linear Shells ---
with tab3:
    st.header("Linear Shells (Coupled)")
    st.markdown("Reference: **Pages 61, 59, 60**")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        Rx = st.number_input("Radius Rx [m] (0 for inf)", value=1.0)
    with col_r2:
        Ry = st.number_input("Radius Ry [m] (0 for inf)", value=0.0) # 0 represents infinity here for logic
        
    k_shear_shell = st.number_input("Shear Correction Factor (k)", value=0.8333, key="k_shell")
    H_q_shell = k_shear_shell * G * h * np.eye(2)

    if st.button("Calculate Shell Resultants"):
        st.markdown("### Step-by-Step Log")
        
        # 1. Lambda
        lam_11 = 1.0/Rx if Rx != 0 else 0.0
        lam_22 = 1.0/Ry if Ry != 0 else 0.0
        st.write(f"**Curvatures (λ):** λ11 = {lam_11:.4f}, λ22 = {lam_22:.4f}")

        # 2. Uncoupled
        n_tilde = H_n @ eps
        m_tilde_prime = H_m @ kap
        st.write(f"**Uncoupled Membrane (ñ):** {n_tilde}")
        st.write(f"**Uncoupled Bending (m̃'):** {m_tilde_prime}")

        # 3. Coupling
        st.markdown("#### Coupling Effects (Eq pg 59-60)")
        
        # N Coupling
        N_shell = np.zeros(3)
        N_shell[0] = n_tilde[0] + lam_11 * m_tilde_prime[0]
        N_shell[1] = n_tilde[1] + lam_22 * m_tilde_prime[1]
        N_shell[2] = n_tilde[2]
        
        # M Coupling
        geom_factor = (h**2) / 12.0
        M_shell = np.zeros(3)
        M_shell[0] = m_tilde_prime[0] + lam_11 * geom_factor * n_tilde[0]
        M_shell[1] = m_tilde_prime[1] + lam_22 * geom_factor * n_tilde[1]
        M_shell[2] = m_tilde_prime[2]

        Q_shell = H_q_shell @ gam_z

        # Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Coupled Forces (N)")
            st.latex(r"\begin{bmatrix} " + f"{N_shell[0]:.4e} \\\\ {N_shell[1]:.4e} \\\\ {N_shell[2]:.4e}" + r" \end{bmatrix}")
        with col2:
            st.subheader("Coupled Moments (M)")
            st.latex(r"\begin{bmatrix} " + f"{M_shell[0]:.4e} \\\\ {M_shell[1]:.4e} \\\\ {M_shell[2]:.4e}" + r" \end{bmatrix}")
        with col3:
            st.subheader("Shear (Q)")
            st.latex(r"\begin{bmatrix} " + f"{Q_shell[0]:.4e} \\\\ {Q_shell[1]:.4e}" + r" \end{bmatrix}")
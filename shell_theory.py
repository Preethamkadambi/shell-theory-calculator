import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np

class ShellTheoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shell Theory Calculator with Detailed Logging")
        self.root.geometry("900x900")

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # --- Tab 1: Reissner-Mindlin Plate (Pages 3-10) ---
        self.tab_rm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rm, text='1. Reissner-Mindlin (pg 3-10)')
        self.create_inputs(self.tab_rm, "rm")

        # --- Tab 2: Kirchhoff-Love Plate (Pages 11-12) ---
        self.tab_kl = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_kl, text='2. Kirchhoff-Love (pg 11-12)')
        self.create_inputs(self.tab_kl, "kl")

        # --- Tab 3: Linear Shells (Pages 61-63) ---
        self.tab_shell = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_shell, text='3. Linear Shells (pg 61)')
        self.create_inputs(self.tab_shell, "shell")

    def create_inputs(self, parent, case_type):
        # Main container using PanedWindow to separate Inputs and Logs
        paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Top Frame for Inputs
        input_frame = ttk.Frame(paned)
        paned.add(input_frame, weight=1)

        # --- Section: Material & Geometry ---
        frame_mat = ttk.LabelFrame(input_frame, text="Material & Geometry Properties")
        frame_mat.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        entries = {}
        
        self.add_entry(frame_mat, "Young's Modulus (E) [Pa]:", "E", entries, "70e9")
        self.add_entry(frame_mat, "Poisson's Ratio (ν):", "nu", entries, "0.3")
        self.add_entry(frame_mat, "Thickness (h_0) [m]:", "h", entries, "0.01")
        
        if case_type in ["rm", "shell"]:
            self.add_entry(frame_mat, "Shear Factor (k) [See Pg 10]:", "k", entries, "0.8333") 

        if case_type == "shell":
            ttk.Label(frame_mat, text="--- Initial Curvature (1/R) ---").pack(pady=5)
            self.add_entry(frame_mat, "Radius Rx [m]:", "Rx", entries, "1.0") 
            self.add_entry(frame_mat, "Radius Ry [m]:", "Ry", entries, "inf") 

        # --- Section: Deformations ---
        frame_strain = ttk.LabelFrame(input_frame, text="Input Generalized Strains")
        frame_strain.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.add_entry(frame_strain, "Membrane ε_11:", "eps_11", entries, "0.001")
        self.add_entry(frame_strain, "Membrane ε_22:", "eps_22", entries, "0.0")
        self.add_entry(frame_strain, "Shear γ_12 (Eng.):", "gam_12", entries, "0.0")

        self.add_entry(frame_strain, "Curvature κ_11 [1/m]:", "kap_11", entries, "0.0")
        self.add_entry(frame_strain, "Curvature κ_22 [1/m]:", "kap_22", entries, "0.5")
        self.add_entry(frame_strain, "Twist κ_12 (Eng.) [1/m]:", "kap_12", entries, "0.0")

        if case_type in ["rm", "shell"]:
            self.add_entry(frame_strain, "Transv. Shear γ_13:", "gam_13", entries, "0.0")
            self.add_entry(frame_strain, "Transv. Shear γ_23:", "gam_23", entries, "0.0")

        # Calculate Button
        btn = ttk.Button(input_frame, text="Calculate & View Detailed Log", 
                         command=lambda: self.calculate(case_type, entries, log_area))
        btn.grid(row=1, column=0, columnspan=2, pady=10)

        # Bottom Frame for Detailed Logging
        log_frame = ttk.LabelFrame(paned, text="Detailed Step-by-Step Calculation Log")
        paned.add(log_frame, weight=3)
        
        log_area = ScrolledText(log_frame, height=20, font=("Consolas", 9), state='disabled')
        log_area.pack(fill="both", expand=True, padx=5, pady=5)

    def add_entry(self, parent, label_text, key, dict_ref, default):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        lbl = ttk.Label(frame, text=label_text, width=25)
        lbl.pack(side="left")
        ent = ttk.Entry(frame, width=15)
        ent.insert(0, default)
        ent.pack(side="right", padx=5)
        dict_ref[key] = ent

    def get_float(self, entry):
        val = entry.get()
        if val == "inf": return np.inf
        try:
            return float(val)
        except ValueError:
            return 0.0

    def format_matrix(self, mat, name):
        """Helper to format numpy arrays for the log."""
        rows = []
        rows.append(f"{name} =")
        for row in mat:
            # Format each number in the row
            fmt_row = [f"{x:.4e}" for x in row]
            rows.append("  [" + ", ".join(fmt_row) + "]")
        return "\n".join(rows) + "\n"

    def log_step(self, text_widget, message):
        text_widget.configure(state='normal')
        text_widget.insert(tk.END, message + "\n")
        text_widget.configure(state='disabled')
        text_widget.see(tk.END)

    def calculate(self, case_type, inputs, log_widget):
        # Clear previous log
        log_widget.configure(state='normal')
        log_widget.delete('1.0', tk.END)
        log_widget.configure(state='disabled')

        try:
            # --- 1. Parse Inputs ---
            self.log_step(log_widget, "=== STEP 1: PARSE INPUTS & CONSTANTS ===")
            E = self.get_float(inputs["E"])
            nu = self.get_float(inputs["nu"])
            h = self.get_float(inputs["h"])
            
            # Derived Constants
            C = E / (1 - nu**2)
            G = E / (2 * (1 + nu))
            
            self.log_step(log_widget, f"Young's Modulus E = {E:.2e} Pa")
            self.log_step(log_widget, f"Poisson's Ratio ν = {nu}")
            self.log_step(log_widget, f"Thickness h_0     = {h} m")
            self.log_step(log_widget, f"Plane Stress Stiffness C = E/(1-ν²) [cite: 122] = {C:.4e} Pa")
            self.log_step(log_widget, f"Shear Modulus G          = E/(2(1+ν))           = {G:.4e} Pa\n")

            # Vectors
            eps = np.array([self.get_float(inputs["eps_11"]), 
                            self.get_float(inputs["eps_22"]), 
                            self.get_float(inputs["gam_12"])])
            
            kap = np.array([self.get_float(inputs["kap_11"]), 
                            self.get_float(inputs["kap_22"]), 
                            self.get_float(inputs["kap_12"])])
            
            self.log_step(log_widget, f"Input Strain Vector ε [11, 22, 12]:   {eps}")
            self.log_step(log_widget, f"Input Curvature Vector κ [11, 22, 12]: {kap}\n")

            # --- 2. Build Stiffness Matrices ---
            self.log_step(log_widget, "=== STEP 2: STIFFNESS MATRICES CALCULATION ===")
            
            # Base Isotropic Matrix (for Voigt notation [11, 22, 12])
            # Ref Page 5 [cite: 122]
            mat_base = np.array([
                [1,      nu,     0],
                [nu,     1,      0],
                [0,      0,      (1-nu)/2] 
            ])
            self.log_step(log_widget, "Constructing Base Material Matrix (Isotropic Plane Stress)...")

            # H_n (Membrane) - Ref Page 5 [cite: 122]
            H_n = C * h * mat_base
            self.log_step(log_widget, f"Calculated Membrane Stiffness H_n (C * h * Base)[cite: 122]:")
            self.log_step(log_widget, self.format_matrix(H_n, "H_n"))

            # H_m (Bending) - Ref Page 5 [cite: 125]
            H_m = C * (h**3 / 12.0) * mat_base
            self.log_step(log_widget, f"Calculated Bending Stiffness H_m (C * h³/12 * Base)[cite: 125]:")
            self.log_step(log_widget, self.format_matrix(H_m, "H_m"))

            # H_q (Shear) - Ref Page 5 [cite: 127]
            H_q = np.zeros((2,2))
            k_shear = 1.0
            if case_type in ["rm", "shell"]:
                k_shear = self.get_float(inputs["k"])
                H_q = k_shear * G * h * np.eye(2)
                self.log_step(log_widget, f"Shear Correction Factor k = {k_shear} [cite: 235]")
                self.log_step(log_widget, f"Calculated Shear Stiffness H_q (k * G * h * I)[cite: 127]:")
                self.log_step(log_widget, self.format_matrix(H_q, "H_q"))
            else:
                self.log_step(log_widget, "Shear Stiffness H_q not applicable for Kirchhoff-Love[cite: 258].\n")

            # --- 3. Resultant Calculation ---
            self.log_step(log_widget, "=== STEP 3: RESULTANT CALCULATION ===")

            if case_type == "kl":
                # Kirchhoff-Love (Page 11-12)
                self.log_step(log_widget, "Method: Kirchhoff-Love (Uncoupled, No Transverse Shear) [cite: 310]")
                N = H_n @ eps
                M = H_m @ kap
                Q = np.zeros(2) # Reactive only
                
                self.log_step(log_widget, "N = H_n * ε")
                self.log_step(log_widget, "M = H_m * κ")
                self.log_step(log_widget, "Q is reactive (not constitutive).\n")

            elif case_type == "rm":
                # Reissner-Mindlin (Page 4-5)
                self.log_step(log_widget, "Method: Reissner-Mindlin (Uncoupled, Includes Shear) [cite: 138, 165]")
                gam_z = np.array([self.get_float(inputs["gam_13"]), self.get_float(inputs["gam_23"])])
                self.log_step(log_widget, f"Input Transverse Shear Strain γ [13, 23]: {gam_z}")
                
                N = H_n @ eps
                M = H_m @ kap
                Q = H_q @ gam_z
                
                self.log_step(log_widget, "N = H_n * ε [cite: 138]")
                self.log_step(log_widget, "M = H_m * κ [cite: 165]")
                self.log_step(log_widget, "Q = H_q * γ [cite: 139]\n")

            elif case_type == "shell":
                # Linear Shell (Page 61, 59, 60)
                self.log_step(log_widget, "Method: Linear Shell (Coupled due to Curvature) [cite: 1545]")
                
                # 3a. Curvature Tensor Approximation
                Rx = self.get_float(inputs["Rx"])
                Ry = self.get_float(inputs["Ry"])
                lam_11 = 1.0/Rx if Rx != 0 else 0.0
                lam_22 = 1.0/Ry if Ry != 0 else 0.0
                self.log_step(log_widget, f"Curvature 1/Rx (λ_11) = {lam_11:.4f}")
                self.log_step(log_widget, f"Curvature 1/Ry (λ_22) = {lam_22:.4f}")
                
                # 3b. Uncoupled Terms
                self.log_step(log_widget, "Calculating Uncoupled Terms (tilde):")
                n_tilde = H_n @ eps
                m_tilde_prime = H_m @ kap
                
                self.log_step(log_widget, f"Uncoupled Membrane ñ = H_n * ε[cite: 2122]: {n_tilde}")
                self.log_step(log_widget, f"Uncoupled Bending m̃' = H_m * κ[cite: 2125]: {m_tilde_prime}\n")

                # 3c. Coupling
                self.log_step(log_widget, "Applying Coupling Terms[cite: 1597, 1537]:")
                
                # N Coupling: n = n_tilde + lambda * m_tilde_prime
                # [cite: 1518] n^ba = n~^ba + lambda_gamma^beta * m~'^alpha_gamma
                # Simplified for principal axes:
                N = np.zeros(3)
                N[0] = n_tilde[0] + lam_11 * m_tilde_prime[0]
                N[1] = n_tilde[1] + lam_22 * m_tilde_prime[1]
                N[2] = n_tilde[2] # Neglecting twist coupling term for simplicity
                
                self.log_step(log_widget, f"Coupling Force N_11 = {n_tilde[0]:.4e} + ({lam_11:.4f} * {m_tilde_prime[0]:.4e})")
                self.log_step(log_widget, f"Coupling Force N_22 = {n_tilde[1]:.4e} + ({lam_22:.4f} * {m_tilde_prime[1]:.4e})")

                # M Coupling: m = m_tilde_prime + lambda * (h^2/12) * n_tilde
                # [cite: 1537] m~ = m~' + lambda * (h^2/12) * n~
                geom_factor = (h**2) / 12.0
                self.log_step(log_widget, f"Geometric Factor (h²/12) = {geom_factor:.4e}")
                
                M = np.zeros(3)
                M[0] = m_tilde_prime[0] + lam_11 * geom_factor * n_tilde[0]
                M[1] = m_tilde_prime[1] + lam_22 * geom_factor * n_tilde[1]
                M[2] = m_tilde_prime[2]

                self.log_step(log_widget, f"Coupling Moment M_11 = {m_tilde_prime[0]:.4e} + ({lam_11:.4f} * {geom_factor:.4e} * {n_tilde[0]:.4e})")
                self.log_step(log_widget, f"Coupling Moment M_22 = {m_tilde_prime[1]:.4e} + ({lam_22:.4f} * {geom_factor:.4e} * {n_tilde[1]:.4e})")

                # Shear
                gam_z = np.array([self.get_float(inputs["gam_13"]), self.get_float(inputs["gam_23"])])
                Q = H_q @ gam_z
                self.log_step(log_widget, f"Shear Q = H_q * γ[cite: 1552]: {Q}\n")

            # --- 4. Final Output ---
            self.log_step(log_widget, "=== STEP 4: FINAL RESULTS ===")
            self.log_step(log_widget, "Membrane Forces (N/m):")
            self.log_step(log_widget, f"  N_11: {N[0]:.4e}")
            self.log_step(log_widget, f"  N_22: {N[1]:.4e}")
            self.log_step(log_widget, f"  N_12: {N[2]:.4e}")
            self.log_step(log_widget, "Bending Moments (N*m/m):")
            self.log_step(log_widget, f"  M_11: {M[0]:.4e}")
            self.log_step(log_widget, f"  M_22: {M[1]:.4e}")
            self.log_step(log_widget, f"  M_12: {M[2]:.4e}")
            if case_type != "kl":
                self.log_step(log_widget, "Transverse Shear Forces (N/m):")
                self.log_step(log_widget, f"  Q_1:  {Q[0]:.4e}")
                self.log_step(log_widget, f"  Q_2:  {Q[1]:.4e}")

            self.log_step(log_widget, "\nCalculation Complete.")

        except Exception as e:
            self.log_step(log_widget, f"\nERROR: {str(e)}")
            messagebox.showerror("Calculation Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ShellTheoryApp(root)
    root.mainloop()
include("../SolutionFunctions.jl")
using Plots
using Statistics
using LaTeXStrings

# Parameters
P = 50.0                 # Choose p so that there are at least three bound states
h = 0.05                 # spatial step size
q = q_of_p(P)
L = 20.0 / q             # domain size
N = round(Int, L / h) - 1
eta = 0.1             # perturbation amplitude (increased to see Rabi oscillations)

# Hanning window function
function hanning_window(N)
    return 0.5 .* (1.0 .- cos.(2π .* (0:N-1) ./ (N-1)))
end

# Step A: Solve the TISE to get eigenstates and eigenvalues
println("Step A: Solving TISE for static potential...")
V_static(xi) = -sech(q * xi)^2
ws = SolverWorkspace(N)
E, U, xi = solve_static_schrodinger(L, h, V_static, ws)

# Store the eigenvalues we need
epsilon_0 = E[1]
epsilon_2 = E[3]
Omega = epsilon_2 - epsilon_0  # resonant frequency

println("Found $(length(E)) bound states")
println("ε₀ = $(epsilon_0)")
println("ε₂ = $(epsilon_2)")
println("Ω = ε₂ - ε₀ = $(Omega)")

# Define time-dependent potential
V_dynamic(xi, tau) = V_static(xi) * (1.0 + eta * sin(Omega * tau))

# Step B: Set initial condition - start in ground state u₀
println("\nStep B: Setting initial condition Ψ(ξ,0) = u₀(ξ)...")

# Normalise all states using L2 norm
U_norm = similar(U)
for i in 1:size(U, 2)
    U_norm[:, i] = normalise_L2(U[:, i], h)
end

psi = ComplexF64.(U_norm[:, 1])  # ground state eigenvector

# Step C: Evolve using Crank-Nicolson
println("\nStep C: Evolving wavefunction with time-dependent potential (LONG TIME)...")
dt = 0.1                  # time step
# Increase simulation time to improve FFT resolution and capture multiple Rabi cycles
T_final = 1024π/Omega              
nsteps = round(Int, T_final / dt)

println("T_final = $T_final (approx $(round(T_final/(2π/Omega), digits=1)) periods)")

# Evolve and project onto basis at each step
# Optimization: only project onto the first few states to save time
n_proj = min(10, size(U_norm, 2))
basis_proj = U_norm[:, 1:n_proj]

psi_evolved, coeffs_history = evolve_dynamic_coeffs!(
    psi, L, h, V_dynamic, dt;
    nsteps=nsteps,
    ws=ws,
    basis=basis_proj,
    store=true
)

# Step D: Analyze coefficients
println("\nStep D: Analyzing state coefficients...")
coeffs_matrix = hcat(coeffs_history...)  # shape: (n_proj, n_timesteps)
magnitudes = abs.(coeffs_matrix)         # |cₙ(τ)|

# Time array scaled by Ω/2π
times = range(0, T_final, length=nsteps+1)
t_scaled = times .* Omega ./ (2 * pi)

# Plot magnitudes of first few states
plt_mag = plot(t_scaled, magnitudes[1, :], label="n=0 (ground)", lw=2, 
    xlabel=L"t\\Omega/(2\\pi)", ylabel=L"|c_n|", title="State Magnitudes vs Scaled Time (Long Run)", dpi=300,
     fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
plot!(plt_mag, t_scaled, magnitudes[2, :], label="n=1", lw=2)
plot!(plt_mag, t_scaled, magnitudes[3, :], label="n=2 (resonant)", lw=2)
if size(magnitudes, 1) >= 4
    plot!(plt_mag, t_scaled, magnitudes[4, :], label="n=3", lw=2)
end
savefig(plt_mag, "Q4/state_magnitudes_long.png")
println("Saved Q4/state_magnitudes_long.png")

# Perform FFT on coefficients of ground and second excited states
# Apply Hanning Window
win = hanning_window(length(coeffs_matrix[1, :]))
coeffs_0_windowed = coeffs_matrix[1, :] .* win
coeffs_2_windowed = coeffs_matrix[3, :] .* win

f, S = discrete_fft(coeffs_0_windowed, dt)
f2, S2 = discrete_fft(coeffs_2_windowed, dt)

plt_fft = plot(f, abs.(S), label=L"n=0 (f \\approx |\\epsilon_0|/2\\pi)", xlabel="Frequency f", ylabel="|FFT|", 
     title="FFT of Coefficients (Long Simulation)", xlim=(0, 0.25), lw=2, dpi=300,
     fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
plot!(plt_fft, f2, abs.(S2), label=L"n=2 (f \\approx |\\epsilon_2|/2\\pi)", lw=2)


savefig(plt_fft, "Q4/fft_coefficients_long.png")
println("Saved Q4/fft_coefficients_long.png")

# --- FFT of Probabilities ---
println("Computing FFT of probabilities...")
probs_0 = abs2.(coeffs_matrix[1, :])
probs_2 = abs2.(coeffs_matrix[3, :])

# Subtract mean to remove DC component and see the oscillation peak
probs_0_ac = probs_0 .- mean(probs_0)
probs_2_ac = probs_2 .- mean(probs_2)

# Apply window to probabilities too
probs_0_ac_win = probs_0_ac .* win
probs_2_ac_win = probs_2_ac .* win

fp, Sp0 = discrete_fft(probs_0_ac_win, dt, norm=:none)
fp2, Sp2 = discrete_fft(probs_2_ac_win, dt, norm=:none)

# Estimate Rabi frequency
# Calculate exact matrix element <2|V_static|0>
# V_pert(t) = eta * V_static * sin(Omega*t)
# Rabi frequency Omega_R = |<2| eta * V_static |0>|
V_vals = V_static.(xi)
# U_norm[:, 1] is ground state (n=0), U_norm[:, 3] is second excited state (n=2)
matrix_element = sum(conj(U_norm[:, 3]) .* V_vals .* U_norm[:, 1]) * h
V_20_exact = abs(matrix_element)
println("Calculated matrix element |<2|V|0>| = $V_20_exact")

Omega_R = eta * V_20_exact
f_Rabi = Omega_R / (2π)
println("Estimated Rabi frequency f_Rabi ≈ $f_Rabi")

plt_prob_fft = plot(fp, abs.(Sp0), label=L"\mathrm{FFT}(|c_0|^2 - \bar{c}_0)", xlabel="Frequency f", ylabel="Magnitude",
                    title="FFT of State Probabilities (Rabi Oscillation)", xlim=(0, 0.01), lw=2, dpi=300,
                    fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
plot!(plt_prob_fft, fp2, abs.(Sp2), label=L"\mathrm{FFT}(|c_2|^2 - \bar{c}_2)", lw=2, ls=:dash)
vline!([f_Rabi], ls=:dot, color=:green, label="Est. Rabi Freq")

savefig(plt_prob_fft, "Q4/fft_probabilities_long.png")
println("Saved Q4/fft_probabilities_long.png")

# --- Zoomed in Plot for Splitting ---
println("Generating zoomed-in plot for Rabi splitting...")
f0_center = abs(epsilon_0) / (2π)
f2_center = abs(epsilon_2) / (2π)
span = 0.003  # Narrow window to see the splitting

# Use zero-padding to increase FFT resolution (interpolation)
# Apply window BEFORE padding
pad_factor = 10
n_pad = length(coeffs_matrix[1, :]) * pad_factor
coeffs_0_padded = [coeffs_0_windowed; zeros(ComplexF64, n_pad)]
coeffs_2_padded = [coeffs_2_windowed; zeros(ComplexF64, n_pad)]

f_pad, S_pad0 = discrete_fft(coeffs_0_padded, dt)
f_pad2, S_pad2 = discrete_fft(coeffs_2_padded, dt)

# Find actual peaks in the window of interest for n=0
idx_window_0 = findall(x -> f0_center - span <= x <= f0_center + span, f_pad)
f_window_0 = f_pad[idx_window_0]
S_window_0 = abs.(S_pad0[idx_window_0])

# Find the two highest peaks
# Simple peak finding: find local maxima
peaks_0_indices = []
for i in 2:length(S_window_0)-1
    if S_window_0[i] > S_window_0[i-1] && S_window_0[i] > S_window_0[i+1]
        push!(peaks_0_indices, i)
    end
end
# Sort by magnitude and take top 2
sort!(peaks_0_indices, by = i -> S_window_0[i], rev=true)
top_peaks_0 = f_window_0[peaks_0_indices[1:2]]
observed_splitting_0 = abs(top_peaks_0[1] - top_peaks_0[2])

println("Observed Splitting (n=0): $observed_splitting_0")
println("Theoretical Splitting: $f_Rabi")

p1 = plot(f_pad, abs.(S_pad0), title=L"n=0\ \mathrm{Splitting\ (Ground)}", xlabel="Frequency", ylabel="|FFT|", 
          xlim=(f0_center - span, f0_center + span), legend=false, lw=2,
          fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
# Mark the OBSERVED peaks
vline!(p1, top_peaks_0, ls=:dot, c=:red, label="Observed")

# Annotate splitting
y_arrow = maximum(S_window_0) * 0.8
plot!(p1, [top_peaks_0[1], top_peaks_0[2]], [y_arrow, y_arrow], arrow=:both, color=:black, lw=1.5)
annotate!(p1, (top_peaks_0[1] + top_peaks_0[2])/2, y_arrow * 1.1, text(L"\\Omega_R / 2\\pi", 10, :bottom))


# Find actual peaks for n=2
idx_window_2 = findall(x -> f2_center - span <= x <= f2_center + span, f_pad2)
f_window_2 = f_pad2[idx_window_2]
S_window_2 = abs.(S_pad2[idx_window_2])

peaks_2_indices = []
for i in 2:length(S_window_2)-1
    if S_window_2[i] > S_window_2[i-1] && S_window_2[i] > S_window_2[i+1]
        push!(peaks_2_indices, i)
    end
end
sort!(peaks_2_indices, by = i -> S_window_2[i], rev=true)
top_peaks_2 = f_window_2[peaks_2_indices[1:2]]
observed_splitting_2 = abs(top_peaks_2[1] - top_peaks_2[2])

p2 = plot(f_pad2, abs.(S_pad2), title=L"n=2\ \mathrm{Splitting\ (Resonant)}", xlabel="Frequency", ylabel="|FFT|", 
          xlim=(f2_center - span, f2_center + span), legend=false, lw=2, color=:green,
          fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
vline!(p2, top_peaks_2, ls=:dot, c=:red)

# Annotate splitting
y_arrow2 = maximum(S_window_2) * 0.8
plot!(p2, [top_peaks_2[1], top_peaks_2[2]], [y_arrow2, y_arrow2], arrow=:both, color=:black, lw=1.5)
annotate!(p2, (top_peaks_2[1] + top_peaks_2[2])/2, y_arrow2 * 1.1, text(L"\\Omega_R / 2\\pi", 10, :bottom))

plt_zoom = plot(p1, p2, layout=(1, 2), size=(1000, 400), dpi=300, margin=5Plots.mm)
savefig(plt_zoom, "Q4/fft_splitting_zoom.png")
println("Saved Q4/fft_splitting_zoom.png")

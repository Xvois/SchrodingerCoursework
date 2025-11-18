# We have our prior work on the Poschl-Teller potential, but now we consider this
# Hamiltonian as the "free" Hamiltonian H0, where are total Hamiltonian is H = H0 + H1(tau)
# with a time-dependent perturbation H1(tau) = eta * sin( |epsilon_2 - epsilon_0| * tau) / cosh^2(q xi)
# where eta is a small parameter controlling the perturbation strength.

# We note that our H1 is harmonic, hence we can apply harmonic time-dependent perturbation theory.

# Putting our H1 in the form H1(tau) = F e^-i*omega*tau + F† e^i*omega*tau, we have
# F = (eta/2im) * (1/cosh^2(q xi)) and omega = |epsilon_2 - epsilon_0|.

include("../SolutionFunctions.jl")

# Parameters
P = 50.0               # dimensionless parameter controlling potential depth
N = 1000               # number of interior grid points
L = 10.0 / q_of_p(P)   # domain size scaled with p
q = q_of_p(P)          # parameter q = 1/sqrt(p)
eta = 0.01             # perturbation strength (small parameter)

# Solve the stationary problem to get eigenvalues and eigenvectors
E, psi, xi = solve_schrodinger(N, L, q)
dx = L / (N + 1)  # grid spacing

# Normalise all eigenvectors with dx-weighted L^2 norm using helper
@views for i in 1:size(psi, 2)
    normalise_L2!(psi[:, i], dx)
end

# Unperturbed energies and states
c0 = project_onto_basis_L2(complex.(psi[:, 3]), psi, dx)  # initial state coefficients (fundamental mode)

@show c0

# We now calculate terms for the first order time-dependent perturbation theory.

# We need the matrix element F_mn = <m|F|n> where F = (eta/2im) * (1/cosh^2(q xi))
F_array = (eta / (2im)) .* (1.0 ./ cosh.(q .* xi).^2)

F1_1 = project_L2(psi[:, 1], F_array .* psi[:, 1], dx)  # <1|F|1>
F1_2 = project_L2(psi[:, 1], F_array .* psi[:, 2], dx)  # <1|F|2>
F1_3 = project_L2(psi[:, 1], F_array .* psi[:, 3], dx)  # <1|F|3>

# Frequency of the perturbation
omega = abs(E[3] - E[1]) # need to de-dimensionalise, epsilon = E / V0 
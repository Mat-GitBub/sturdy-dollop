import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

mp.mp.dps = 32  # set precision (digits)


def chebyshev_points(n):
    pts = []
    for k in range(1, n + 2):
        val = mp.cos((2 * k - 1) * mp.pi / (2 * n + 2))
        pts.append(val)
    return list(reversed(pts))


def chebyshev_second_points(n):
    pts = []
    for k in range(1, n + 2):
        val = mp.cos(k * mp.pi / (n + 2))
        pts.append(val)
    return list(reversed(pts))


def compute_barycentric_weights(x_nodes):
    """
    compute the barycentric weights for interpolation nodes
    """
    n = len(x_nodes)
    w = [mp.mpf(1) for _ in range(n)]

    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (x_nodes[j] - x_nodes[k])
    return w


def lebesgue_function(x_nodes, x_eval):
    """
    Compute the Lebesgue function over x_eval using barycentric formula.
    """
    w = compute_barycentric_weights(x_nodes)
    lebesgue_vals = []

    for x in x_eval:
        diffs = [x - xn for xn in x_nodes]

        # Handle exact node match
        if any(d == 0 for d in diffs):
            lebesgue_vals.append(mp.mpf(1))
        else:
            terms = [abs(w[j] / diffs[j]) for j in range(len(x_nodes))]
            numerator = mp.fsum(terms)
            denominator = abs(mp.fsum([w[j] / diffs[j] for j in range(len(x_nodes))]))
            lebesgue_vals.append(numerator / denominator)

    return lebesgue_vals


# Parameters
deg = 36
# nodes = chebyshev_second_points(deg)
nodes = chebyshev_points(deg)
# nodes = [-1+2*(j-1)/deg for j in range(1,deg+2)]
# nodes = [-1, -0.2, 0, 0.2, 1]
print(nodes)
# The interval over which we combute finite differences
Interval_away_from_edges = 2
# number of points for evaluation
eval_points = 20001



interval = [-mp.mpf("0.0"), mp.mpf("1.0")]
interval2 = [nodes[-Interval_away_from_edges], nodes[-Interval_away_from_edges]+0.0006*(nodes[-Interval_away_from_edges+1]-nodes[-Interval_away_from_edges])]
# interval2 = [nodes[-2], nodes[-1]]


connecting_line_x = [nodes[-2], 0.991902228222]
y_val = lebesgue_function(nodes, [0.991902228222])
print(nodes[-2] - 0.991902228222)
connecting_line_y = [1, *y_val]
print(connecting_line_y)

# base-list version of linspace
def mp_linspace(a, b, n):
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

x_eval = mp_linspace(interval[0], interval[1], eval_points)
x_eval2 = mp_linspace(interval2[0], interval2[1], eval_points)

lebesgue_vals = lebesgue_function(nodes, x_eval)
lebesgue_vals2 = lebesgue_function(nodes, x_eval2)


Num_diff = []
for index in range(1, len(x_eval2)-1):
    Num_diff.append(lebesgue_vals2[index-1] - 2*lebesgue_vals2[index] + lebesgue_vals2[index+1])

sign_change_points = []
for i in range(1, len(Num_diff)):
    prev_val = Num_diff[i - 1]
    curr_val = Num_diff[i]

    # Detect sign change
    if prev_val == 0:
        continue
    if (prev_val > 0 and curr_val < 0) or (prev_val < 0 and curr_val > 0):
        # Map back to x index
        x_index = i + 1
        sign_change_points.append(x_eval2[x_index])

# Print results
for x in sign_change_points:
    print(x)

if len(sign_change_points) == 1:
    print((nodes[-Interval_away_from_edges], sign_change_points[0]))

##############################
######     Ploting     #######
##############################

Num_diff_plot = [float(x) for x in Num_diff]

plt.figure(figsize=(8, 4))
plt.plot(x_eval2[1:-1], Num_diff)
plt.plot(x_eval2[1:-1], np.zeros_like(x_eval2[1:-1]))

#######################################################################################

# Convert to float for plotting (matplotlib doesn't handle mpf well)
x_plot = [float(x) for x in x_eval2]
y_plot = [float(y) for y in lebesgue_vals2]

plt.figure(figsize=(8, 4))
plt.plot(x_plot, y_plot, label='Lebesgue Function of the Chebyshev points (I type)')
plt.plot(connecting_line_x, connecting_line_y, c='r', label='Line connecting points (xₙ, λ(xₙ)) and (xₙ + 1.8·10⁻⁶, λ(xₙ + 1.8·10⁻⁶))')

plt.legend()


plt.grid(True)
plt.show()
#######################################################################################

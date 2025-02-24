# Network-Traffic-Signal-Tampering

Intersection capacities can influence local traffic flows, which can impact traffic conditions throughout an urban network. It has been demonstrated that signal control infrastructure is susceptible to cyber-attacks. This paper develops an optimization model of signal tampering. The attack is modeled as a bi-objective optimization problem, where we simultaneously seek to reduce vehicular throughput in
the network over time (maximize impact) while introducing minimal changes to network signal timings (minimize noticeability). We represent the Spatio-temporal traffic dynamics as a static network flow problem on a time-expanded graph. This allows us to reduce the (non-convex) attack problem to a tractable form, which can be solved with polynomial complexity. We show that minor but objective adjustments in the
signal timings over time can severely impact traffic conditions at the network level. We investigate network vulnerability by examining the concavity of the Pareto-optimal frontier obtained by solving the bi-objective attack problem. Numerical experiments are included to investigate the model and typical results that would be obtained using four toy networks.

![Vulnerability measure](./Net-struct-comp-lowdemand-norm.pdf)

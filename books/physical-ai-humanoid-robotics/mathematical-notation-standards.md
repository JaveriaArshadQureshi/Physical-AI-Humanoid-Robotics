# Mathematical Notation Standards for Physical AI & Humanoid Robotics Book

## Vectors and Matrices

- Vectors: **v** (bold lowercase) or v⃗ (arrow notation)
- Matrices: **M** (bold uppercase)
- Identity matrix: **I**
- Zero vector/matrix: **0**

## Coordinate Systems

- World coordinate frame: W
- Robot base frame: B
- End-effector frame: E
- Camera frame: C
- Joint i frame: i

## Transformations

- Homogeneous transformation from frame A to frame B: **T**<sub>AB</sub>
- Rotation matrix from frame A to frame B: **R**<sub>AB</sub>
- Translation vector: **p**<sub>AB</sub>

## Kinematics

- Joint angles: θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>
- Joint velocities: θ̇<sub>1</sub>, θ̇<sub>2</sub>, ..., θ̇<sub>n</sub>
- Forward kinematics: **x** = f(θ)
- Jacobian matrix: **J**(θ)
- Inverse kinematics: θ = f<sup>-1</sup>(**x**)

## Dynamics

- Mass matrix: **M**(q)
- Coriolis matrix: **C**(q, q̇)
- Gravity vector: **g**(q)
- External forces: τ

## Control

- Desired trajectory: x<sub>d</sub>, ẋ<sub>d</sub>, ẍ<sub>d</sub>
- Error: e = x<sub>d</sub> - x
- Proportional gain: **K**<sub>p</sub>
- Derivative gain: **K**<sub>d</sub>
- Integral gain: **K**<sub>i</sub>

## Probabilities and Statistics

- Probability: P(A)
- Conditional probability: P(A|B)
- Expected value: E[X]
- Variance: Var(X) or σ²
- Covariance matrix: Σ

## Time Notation

- Continuous time: t
- Discrete time: k or t<sub>k</sub>
- Derivative with respect to time: ẋ = dx/dt
- Second derivative with respect to time: ẍ = d²x/dt²
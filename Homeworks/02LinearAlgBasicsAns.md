# Linear Algebra Basics

## Vector Dot Product

Assume $A = \begin{bmatrix}1\\2\\2\end{bmatrix}B = \begin{bmatrix}2\\-1\\2\end{bmatrix}C = \begin{bmatrix}0\\2\\0\end{bmatrix}D = \begin{bmatrix}2\\2\\-1\end{bmatrix}$

1. Which pairs of vectors have an acute angle between them?  Which have an obtuse angle?  Are any perpedicular? **Acute:AB, AC, AD, CD. Obtuse: BC. Right: BD**

2. What is the angle between A and B? Between B and C? 

**Ans: $AB = cos^{-1}(4/(3 * 3)) = 1.11R, 63.6\degree; CD = cos^{-1}(-2/(3*2)) = 1.91R, 109.5\degree$**

## Matrix Multiplication

Assume $A = \begin{bmatrix}1&-1&0\\1&1&0\\0&0&1.41\end{bmatrix}C = \begin{bmatrix}.8\\0\\.6\end{bmatrix}D = \begin{bmatrix}8\\2\\1\end{bmatrix}$

3. What, geometrically, does multiplication by A accomplish?  Answer in terms of rotations and expansion/contraction, without any specific reference to the elements of A.  **Rotates ccw around Z axis, while expanding by $\sqrt{2}$ in all dimensions**

4. What is the perpendicular projection of D in the direction of C?  (Note C is a unit vector, which helps) **DC = 7, so projection is [5.6, 0, 4.2]**

5. Fill in matrix B so that multiplying any vector by it, maps that vectors perpendicularly into the direction of vector C.  So, for instance, BD would map D into the result you had for the prior problem, for instance.  Hint: Consider how the i, j, and k vectors change when mapped onto C.

$B = \begin{bmatrix}.&.&.\\.&.&.\\.&.&.\end{bmatrix}$

**Ans: $B = \begin{bmatrix}.64&0&.48\\0&0&0\\.48&0&.36\end{bmatrix}$**


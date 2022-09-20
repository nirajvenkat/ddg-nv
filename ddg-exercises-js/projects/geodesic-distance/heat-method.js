"use strict";

class HeatMethod {
	/**
	 * This class implements the {@link http://cs.cmu.edu/~kmcrane/Projects/HeatMethod/ heat method} to compute geodesic distance
	 * on a surface mesh.
	 * @constructor module:Projects.HeatMethod
	 * @param {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {Object} vertexIndex A dictionary mapping each vertex of the input mesh to a unique index.
	 * @property {module:LinearAlgebra.SparseMatrix} A The laplace matrix of the input mesh.
	 * @property {module:LinearAlgebra.SparseMatrix} F The mean curvature flow operator built on the input mesh.
	 */
	constructor(geometry) {
		this.geometry = geometry;
		this.vertexIndex = indexElements(geometry.mesh.vertices);

		// We use the time step t = h^2
		let h = this.geometry.meanEdgeLength();
		let M = this.geometry.massMatrix(this.vertexIndex);

		// Build Laplace and flow matrices
		this.A = this.geometry.laplaceMatrix(this.vertexIndex);
		this.F = M.plus(this.A.timesReal(h * h));
	}

	/**
	 * Computes the vector field X = -∇u / |∇u|.
	 * @private
	 * @method module:Projects.HeatMethod#computeVectorField
	 * @param {module:LinearAlgebra.DenseMatrix} u A dense vector (i.e., u.nCols() == 1) representing the
	 * heat that is allowed to diffuse on the input mesh for a brief period of time.
	 * @returns {Object} A dictionary mapping each face of the input mesh to a {@link module:LinearAlgebra.Vector Vector}.
	 */
	computeVectorField(u) {
		// Using the method for simplicial meshes in Section 3.2.1
		let X = {};

		for (let f of this.geometry.mesh.faces) {
			const area = this.geometry.area(f);
			const N = this.geometry.faceNormal(f);
			let gradU = new Vector();

			for (let h of f.adjacentHalfedges()) {
				// Vertex opposite this halfedge
				const vIdx = this.vertexIndex[h.corner.vertex];
				// Heat at the opposite vertex
				const u_i = u.get(vIdx, 0);
				let e_i = this.geometry.vector(h);
				let cross = N.cross(e_i);
				cross.scaleBy(u_i);

				gradU.incrementBy(cross);
			}

			// Gradient of this triangle
			gradU.divideBy(2.0 * area);
			gradU.normalize();
			X[f] = gradU.negated();
		}

		return X;
	}

	/**
	 * Computes the integrated divergence ∇.X.
	 * @private
	 * @method module:Projects.HeatMethod#computeDivergence
	 * @param {Object} X The vector field -∇u / |∇u| represented by a dictionary
	 * mapping each face of the input mesh to a {@link module:LinearAlgebra.Vector Vector}.
	 * @returns {module:LinearAlgebra.DenseMatrix}
	 */
	computeDivergence(X) {
		let V = this.geometry.mesh.vertices.length;
		let divX = DenseMatrix.zeros(V, 1);

		for (let v of this.geometry.mesh.vertices) {
			let vIdx = this.vertexIndex[v];
			let sum = 0.0;

			for (let h of v.adjacentHalfedges()) {
				let e_1 = this.geometry.vector(h);
				let e_2 = this.geometry.vector(h.prev).negated();
				let cot_1 = this.geometry.cotan(h);
				let cot_2 = this.geometry.cotan(h.prev);
				// Vector field at this face
				const X_j = X[h.face];

				sum += cot_1 * e_1.dot(X_j) + cot_2 * e_2.dot(X_j);
			}

			// Divergence at this vertex
			divX.set(0.5 * sum, vIdx, 0);
		}

		return divX;
	}

	/**
	 * Shifts φ such that its minimum value is zero.
	 * @private
	 * @method module:Projects.HeatMethod#subtractMinimumDistance
	 * @param {module:LinearAlgebra.DenseMatrix} phi The (minimum 0) solution to the poisson equation Δφ = ∇.X.
	 */
	subtractMinimumDistance(phi) {
		let min = Infinity;
		for (let i = 0; i < phi.nRows(); i++) {
			min = Math.min(phi.get(i, 0), min);
		}

		for (let i = 0; i < phi.nRows(); i++) {
			phi.set(phi.get(i, 0) - min, i, 0);
		}
	}

	/**
	 * Computes the geodesic distances φ using the heat method.
	 * @method module:Projects.HeatMethod#compute
	 * @param {module:LinearAlgebra.DenseMatrix} delta A dense vector (i.e., delta.nCols() == 1) containing
	 * heat sources, i.e., u0 = δ(x).
	 * @returns {module:LinearAlgebra.DenseMatrix}
	 */
	compute(delta) {
		// Integrate heat flow
		let llt = this.F.chol();
		let u = llt.solvePositiveDefinite(delta);

		// Compute unit vector field X and divergence ∇.X
		let X = this.computeVectorField(u);
		let div = this.computeDivergence(X);

		// Solve poisson equation Δφ = ∇.X
		llt = this.A.chol();
		let phi = llt.solvePositiveDefinite(div.negated());

		// Since φ is unique up to an additive constant, it should
		// be shifted such that the smallest distance is zero
		this.subtractMinimumDistance(phi);

		return phi;
	}
}

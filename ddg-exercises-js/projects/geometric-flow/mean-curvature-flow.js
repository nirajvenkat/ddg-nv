"use strict";

class MeanCurvatureFlow {
	/**
	 * This class performs {@link https://cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf mean curvature flow} on a surface mesh.
	 * @constructor module:Projects.MeanCurvatureFlow
	 * @param {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {Object} vertexIndex A dictionary mapping each vertex of the input mesh to a unique index.
	 */
	constructor(geometry) {
		this.geometry = geometry;
		this.vertexIndex = indexElements(geometry.mesh.vertices);
	}

	/**
	 * Builds the mean curvature flow operator.
	 * @private
	 * @method module:Projects.MeanCurvatureFlow#buildFlowOperator
	 * @param {module:LinearAlgebra.SparseMatrix} M The mass matrix of the input mesh.
	 * @param {number} h The timestep.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	buildFlowOperator(M, h) {
		let A = this.geometry.laplaceMatrix(this.vertexIndex);

		// Flow operator computed using Forward Euler update: F = M + hA
		return M.plus(A.timesReal(h));
	}

	/**
	 * Performs mean curvature flow on the input mesh with timestep h.
	 * @method module:Projects.MeanCurvatureFlow#integrate
	 * @param {number} h The timestep.
	 */
	integrate(h) {
		let vertices = this.geometry.mesh.vertices;
		let V = vertices.length;
		let M = this.geometry.massMatrix(this.vertexIndex);
		let F = this.buildFlowOperator(M, h);

		// Encode the surface as a single matrix of size |V| * 3
		let b = DenseMatrix.zeros(V, 3);
		for (let v of vertices) {
			let i = this.vertexIndex[v];
			let p = this.geometry.positions[v];

			b.set(p.x, i, 0);
			b.set(p.y, i, 1);
			b.set(p.z, i, 2);
		}
		b = M.timesDense(b);

		// Solve Fx = b using Cholesky solver
		let llt = F.chol();
		let x = llt.solvePositiveDefinite(b);

		// Update vertices using the solution x
		for (let v of vertices) {
			let i = this.vertexIndex[v];
			let p = this.geometry.positions[v];

			p.x = x.get(i, 0);
			p.y = x.get(i, 1);
			p.z = x.get(i, 2);
		}

		// Center mesh positions around origin
		normalize(this.geometry.positions, vertices, false);
	}
}

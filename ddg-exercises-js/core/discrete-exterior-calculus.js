"use strict";

/**
 * This class contains methods to build common {@link https://cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf discrete exterior calculus} operators.
 * @memberof module:Core
 */
class DEC {
	/**
	 * Builds a sparse diagonal matrix encoding the Hodge operator on 0-forms.
	 * By convention, the area of a vertex is 1.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} vertexIndex A dictionary mapping each vertex of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildHodgeStar0Form(geometry, vertexIndex) {
		let mesh = geometry.mesh;
		let V = mesh.vertices.length;
		let T = new Triplet(V, V);

		mesh.vertices.forEach((vertex, vIdx) => {
			// We treat volume of a vertex as 1
			let volumeRatio = geometry.barycentricDualArea(vertex);
			T.addEntry(volumeRatio, vIdx, vIdx);
		});

		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse diagonal matrix encoding the Hodge operator on 1-forms.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} edgeIndex A dictionary mapping each edge of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildHodgeStar1Form(geometry, edgeIndex) {
		let mesh = geometry.mesh;
		let E = mesh.edges.length;
		let T = new Triplet(E, E);

		mesh.edges.forEach((edge, eIdx) => {
			let h1 = edge.halfedge;
			let h2 = edge.halfedge.twin;
			// Ratio of dual to primal edge lengths using the cotan formula
			let volumeRatio = 0.5 * (geometry.cotan(h1) + geometry.cotan(h2));
			T.addEntry(volumeRatio, eIdx, eIdx);
		});

		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse diagonal matrix encoding the Hodge operator on 2-forms.
	 * By convention, the area of a vertex is 1.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} faceIndex A dictionary mapping each face of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildHodgeStar2Form(geometry, faceIndex) {
		let mesh = geometry.mesh;
		let F = mesh.faces.length;
		let T = new Triplet(F, F);

		mesh.faces.forEach((face, fIdx) => {
			// We treat volume of a vertex as 1
			let volumeRatio = 1.0 / geometry.area(face);
			T.addEntry(volumeRatio, fIdx, fIdx);
		});

		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse matrix encoding the exterior derivative on 0-forms.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} edgeIndex A dictionary mapping each edge of a mesh to a unique index.
	 * @param {Object} vertexIndex A dictionary mapping each vertex of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildExteriorDerivative0Form(geometry, edgeIndex, vertexIndex) {
		let mesh = geometry.mesh;
		let T = new Triplet(mesh.edges.length, mesh.vertices.length);

		mesh.edges.forEach(edge => {
			const {vertex: v1, twin: {vertex: v2}} = edge.halfedge;
			T.addEntry(1, edge.index, v1.index);
			T.addEntry(-1, edge.index, v2.index);
		});

		return SparseMatrix.fromTriplet(T);
	}

	/**
	 * Builds a sparse matrix encoding the exterior derivative on 1-forms.
	 * @static
	 * @param {module:Core.Geometry} geometry The geometry of a mesh.
	 * @param {Object} faceIndex A dictionary mapping each face of a mesh to a unique index.
	 * @param {Object} edgeIndex A dictionary mapping each edge of a mesh to a unique index.
	 * @returns {module:LinearAlgebra.SparseMatrix}
	 */
	static buildExteriorDerivative1Form(geometry, faceIndex, edgeIndex) {
		let mesh = geometry.mesh;
		let T = new Triplet(mesh.faces.length, mesh.edges.length);

		mesh.faces.forEach(face => {
			let fIdx = faceIndex[face];
			for (let h of face.adjacentHalfedges()) {
				let eIdx = edgeIndex[h.edge];
				let sign = h.edge.halfedge === h ? 1 : -1;
				T.addEntry(sign, fIdx, eIdx);
			}
		});

		return SparseMatrix.fromTriplet(T);
	}
}
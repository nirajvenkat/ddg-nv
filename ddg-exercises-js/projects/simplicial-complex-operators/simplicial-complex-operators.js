"use strict";

/**
 * @module Projects
 */
class SimplicialComplexOperators {

		/** This class implements various operators (e.g. boundary, star, link) on a mesh.
		 * @constructor module:Projects.SimplicialComplexOperators
		 * @param {module:Core.Mesh} mesh The input mesh this class acts on.
		 * @property {module:Core.Mesh} mesh The input mesh this class acts on.
		 * @property {module:LinearAlgebra.SparseMatrix} A0 The vertex-edge adjacency matrix of <code>mesh</code>.
		 * @property {module:LinearAlgebra.SparseMatrix} A1 The edge-face adjacency matrix of <code>mesh</code>.
		 */
		constructor(mesh) {
			this.mesh = mesh;
			this.assignElementIndices(this.mesh);

			this.A0 = this.buildVertexEdgeAdjacencyMatrix(this.mesh);
			this.A1 = this.buildEdgeFaceAdjacencyMatrix(this.mesh);
		}

		/** Assigns indices to the input mesh's vertices, edges, and faces
		 * @method module:Projects.SimplicialComplexOperators#assignElementIndices
		 * @param {module:Core.Mesh} mesh The input mesh which we index.
		 */
		assignElementIndices(mesh) {
			mesh.vertices.forEach((vertex, idx) => vertex.index = idx);
			mesh.edges.forEach((edge, idx) => edge.index = idx);
			mesh.faces.forEach((face, idx) => face.index = idx); 
		}

		/** Returns the vertex-edge adjacency matrix of the given mesh.
		 * @method module:Projects.SimplicialComplexOperators#buildVertexEdgeAdjacencyMatrix
		 * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
		 * @returns {module:LinearAlgebra.SparseMatrix} The vertex-edge adjacency matrix of the given mesh.
		 */
		buildVertexEdgeAdjacencyMatrix(mesh) {
			let T = new Triplet(mesh.edges.length, mesh.vertices.length);

			mesh.edges.forEach(edge => {
				const {vertex: v1, twin: {vertex: v2}} = edge.halfedge;
				T.addEntry(1, edge.index, v1.index);
				T.addEntry(1, edge.index, v2.index);
			});

			return SparseMatrix.fromTriplet(T);
		}

		/** Returns the edge-face adjacency matrix.
		 * @method module:Projects.SimplicialComplexOperators#buildEdgeFaceAdjacencyMatrix
		 * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
		 * @returns {module:LinearAlgebra.SparseMatrix} The edge-face adjacency matrix of the given mesh.
		 */
		buildEdgeFaceAdjacencyMatrix(mesh) {
			let T = new Triplet(mesh.faces.length, mesh.edges.length);

			mesh.faces.forEach(face => {
				const {edge: e1, next: {edge: e2, next: {edge: e3}}} = face.halfedge;
				T.addEntry(1, face.index, e1.index);
				T.addEntry(1, face.index, e2.index);
				T.addEntry(1, face.index, e3.index);
			});

			return SparseMatrix.fromTriplet(T);
		}

		/** Returns a column vector representing the vertices of the
		 * given subset.
		 * @method module:Projects.SimplicialComplexOperators#buildVertexVector
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |V| entries. The ith entry is 1 if
		 *  vertex i is in the given subset and 0 otherwise
		 */
		buildVertexVector(subset) {
			let M = DenseMatrix.zeros(this.mesh.vertices.length, 1);
			subset.vertices.forEach(vIdx => M.set(1, vIdx, 0));
			return M;
		}

		/** Returns a column vector representing the edges of the
		 * given subset.
		 * @method module:Projects.SimplicialComplexOperators#buildEdgeVector
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |E| entries. The ith entry is 1 if
		 *  edge i is in the given subset and 0 otherwise
		 */
		buildEdgeVector(subset) {
			let M = DenseMatrix.zeros(this.mesh.edges.length, 1);
			subset.edges.forEach(eIdx => M.set(1, eIdx, 0));
			return M;
		}

		/** Returns a column vector representing the faces of the
		 * given subset.
		 * @method module:Projects.SimplicialComplexOperators#buildFaceVector
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |F| entries. The ith entry is 1 if
		 *  face i is in the given subset and 0 otherwise
		 */
		buildFaceVector(subset) {
			let M = DenseMatrix.zeros(this.mesh.faces.length, 1);
			subset.faces.forEach(fIdx => M.set(1, fIdx, 0));
			return M;
		}

		/** Returns the star of a subset.
		 * @method module:Projects.SimplicialComplexOperators#star
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {module:Core.MeshSubset} The star of the given subset.
		 */
		star(subset) {
			let star = MeshSubset.deepCopy(subset);

			// Add edges to star
			star.vertices.forEach(vertex => {
				let tempSubset = new MeshSubset();
				tempSubset.addVertex(vertex);
				const oneHotVertex = this.buildVertexVector(tempSubset);
				const connEdges = this.A0.timesDense(oneHotVertex);
				for (let row = 0; row < connEdges.nRows(); row++) {
					const isConnEdge = connEdges.get(row, 0);
					if (isConnEdge) {
						star.addEdge(row);
					}
				}
			});

			// Add faces to star
			star.edges.forEach(edge => {
				let tempSubset = new MeshSubset();
				tempSubset.addEdge(edge);
				const oneHotEdge = this.buildEdgeVector(tempSubset);
				const connFaces = this.A1.timesDense(oneHotEdge);
				for (let row = 0; row < connFaces.nRows(); row++) {
					const isConnFace = connFaces.get(row, 0);
					if (isConnFace) {
						star.addFace(row);
					}
				}
			});
			
			return star;
		}

		/** Returns the closure of a subset.
		 * @method module:Projects.SimplicialComplexOperators#closure
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {module:Core.MeshSubset} The closure of the given subset.
		 */
		closure(subset) {
			let closure = MeshSubset.deepCopy(subset);

			// Add edges to closure
			closure.faces.forEach(face => {
				let tempSubset = new MeshSubset();
				tempSubset.addFace(face);
				const oneHotFace = this.buildFaceVector(tempSubset);
				const connEdges = this.A1.transpose().timesDense(oneHotFace);
				for (let row = 0; row < connEdges.nRows(); row++) {
					const isConnEdge = connEdges.get(row, 0);
					if (isConnEdge) {
						closure.addEdge(row);
					}
				}
			});

			// Add vertices to closure
			closure.edges.forEach(edge => {
				let tempSubset = new MeshSubset();
				tempSubset.addEdge(edge);
				const oneHotEdge = this.buildEdgeVector(tempSubset);
				const connVertices = this.A0.transpose().timesDense(oneHotEdge);
				for (let row = 0; row < connVertices.nRows(); row++) {
					const isConnVertex = connVertices.get(row, 0);
					if (isConnVertex) {
						closure.addVertex(row);
					}
				}
			});

			return closure;
		}

		/** Returns the link of a subset.
		 * @method module:Projects.SimplicialComplexOperators#link
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {module:Core.MeshSubset} The link of the given subset.
		 */
		link(subset) {
			// The link Lk(S) is equal to Cl(St(S)) \ St(Cl(S)).
			let clSt = this.closure(this.star(subset));
			let stCl = this.star(this.closure(subset));                
			clSt.deleteSubset(stCl);
			return clSt; 
		}

		/** Returns true if the given subset is a subcomplex and false otherwise.
		 * @method module:Projects.SimplicialComplexOperators#isComplex
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {boolean} True if the given subset is a subcomplex and false otherwise.
		 */
		isComplex(subset) {
			return this.closure(subset).equals(subset);
		}

		/** Returns the degree if the given subset is a pure subcomplex and -1 otherwise.
		 * @method module:Projects.SimplicialComplexOperators#isPureComplex
		 * @param {module:Core.MeshSubset} subset A subset of our mesh.
		 * @returns {number} The degree of the given subset if it is a pure subcomplex and -1 otherwise.
		 */
		isPureComplex(subset) {
			let degree = 0;

			let facesSize = subset.faces.size;
			let subsetEdges = this.buildEdgeVector(subset);
			let faceEdgeVector = this.A1.transpose().timesDense(this.buildFaceVector(subset));
			if (facesSize > 0) {
				for (let row = 0; row < subsetEdges.nRows(); row++) {
					let se = subsetEdges.get(row, 0);
					let fe = faceEdgeVector.get(row, 0);
					if (((se != 0) != (fe != 0))) return -1;
				}
				degree = 2;
			}

			let edgesSize = subset.edges.size;
			let subsetVertices = this.buildVertexVector(subset);
			let edgeVertexVector = this.A0.transpose().timesDense(this.buildEdgeVector(subset));
			if (edgesSize > 0) {
				for (let row = 0; row < subsetVertices.nRows(); row++) {  
					let sv = subsetVertices.get(row, 0);
					let ev = edgeVertexVector.get(row, 0);
					if (((sv != 0) != (ev != 0))) return -1;
				}
				if (degree != 2) degree = 1;
			}

			return degree;
		}

		/** Returns the boundary of a subset.
		 * @method module:Projects.SimplicialComplexOperators#boundary
		 * @param {module:Core.MeshSubset} subset A subset of our mesh. We assume <code>subset</code> is a pure subcomplex.
		 * @returns {module:Core.MeshSubset} The boundary of the given pure subcomplex.
		 */
		boundary(subset) {
			let boundary = new MeshSubset();
			if (subset.faces.size > 0) {
				let faceEdges = this.A1.transpose().timesDense(this.buildFaceVector(subset));
				for (let i = 0; i < this.mesh.edges.length; i++) {
					if (faceEdges.get(i, 0) == 1) {
						boundary.addEdge(i);
					}
				}
			} else if (subset.edges.size > 0) {
				let edgeVertices = this.A0.transpose().timesDense(this.buildEdgeVector(subset));
				for (let i = 0; i < this.mesh.vertices.length; i++) {
					if (edgeVertices.get(i, 0) == 1) {
						boundary.addVertex(i);
					}
				}
			}
			return this.closure(boundary);
		}
}

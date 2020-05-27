#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Utils/getopt.h>
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <mkl.h>
#include <mkl_sparse_qr.h>


using namespace std;

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;


// calculate [v2 - v1, v3 - v1, v4 - v1] for any face
inline void calculateV(const MyMesh &mesh, const OpenMesh::FaceHandle &hface, Eigen::Matrix3f &outMatrix) {
	OpenMesh::Vec3f v[4];
	int i = 0;

	// read 3 vertice of the face to v[0-2]
	for (auto it = mesh.cfv_begin(hface); it != mesh.cfv_end(hface); ++it) {
		v[i++] = mesh.point(*it);
	}

	v[1] -= v[0];					// v2 - v1
	v[2] -= v[0];					// v3 - v1
	v[3] = v[1] % v[2];
	v[3] /= sqrt(v[3].length());	// v4 - v1

	outMatrix << v[1].data()[0], v[2].data()[0], v[3].data()[0],
				 v[1].data()[1], v[2].data()[1], v[3].data()[1],
				 v[1].data()[2], v[2].data()[2], v[3].data()[2];
}


// change t0 to t1 according to s0 -> s1, output t1
void deformation_transfer(const char *const s0_obj_path, const char *const s1_obj_path,
						  const char *const t0_obj_path, const char *const t1_obj_path) {

	MyMesh meshS0, meshS1, meshT0, meshT1;

	// read mesh
	#pragma omp parallel sections num_threads(3)
	{
		#pragma omp section
		if (!OpenMesh::IO::read_mesh(meshS0, s0_obj_path)) exit(1);
		#pragma omp section
		if (!OpenMesh::IO::read_mesh(meshS1, s1_obj_path)) exit(1); 
		#pragma omp section
		if (!OpenMesh::IO::read_mesh(meshT0, t0_obj_path)) exit(1); 
	}

	// allocate b, col-major 
	const size_t faceNum = meshS0.n_faces();
	const size_t verticeNum = meshS0.n_vertices();
	const size_t rownum = faceNum * 3 + 1;
	const size_t colnum = faceNum + verticeNum;
	const size_t nznum = faceNum * 6 + 1;		// number of non-zero elements in A
	double *const bb = new double[rownum * 3];
	double *const b[3] = { bb, bb + rownum, bb + 2 * rownum };

	// setting matrix A (rectangular sparse general matrix), CSR
	sparse_status_t status = SPARSE_STATUS_SUCCESS;
	sparse_matrix_t csrA = NULL;
	struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
	MKL_INT *const ia = new MKL_INT[rownum + 1];
	MKL_INT *const ja = new MKL_INT[nznum], *pja = ja;
	double *const a = new double[nznum];


	Eigen::Index rowidx = 0;
	bool firstLoop = true;
	Eigen::Vector3f vt0n;	// v0 of T1
	int idxvt0n;			// index of v0 of T1

	// traverse all faces
	for (auto itFaceS0 = meshS0.faces_begin(),
		 itFaceS1 = meshS1.faces_begin(),
		 itFaceT0 = meshT0.faces_begin();
		 itFaceS0 != meshS0.faces_end(); ++itFaceS0, ++itFaceS1, ++itFaceT0) {

		// calculate Vs, Vsn, Vt
		Eigen::Matrix3f Vs, Vsn, Vt, Vtn;
		calculateV(meshS0, *itFaceS0, Vs);
		calculateV(meshS1, *itFaceS1, Vsn);
		calculateV(meshT0, *itFaceT0, Vt);
		Vtn = Vsn * Vs.inverse() * Vt;

		// first loop calculate global offset and set v0 for T1
		if (firstLoop) {
			OpenMesh::Vec3f meshVs0 = meshS0.point(*meshS0.cfv_begin(*itFaceS0));
			OpenMesh::Vec3f meshVs0n = meshS1.point(*meshS1.cfv_begin(*itFaceS1));
			OpenMesh::Vec3f meshVt0 = meshT0.point(*meshT0.cfv_begin(*itFaceT0));

			Eigen::Vector3f vs0(meshVs0.data()[0], meshVs0.data()[1], meshVs0.data()[2]);
			Eigen::Vector3f vs0n(meshVs0n.data()[0], meshVs0n.data()[1], meshVs0n.data()[2]);
			Eigen::Vector3f vt0(meshVt0.data()[0], meshVt0.data()[1], meshVt0.data()[2]);
			vt0n = vs0n + Vsn * Vs.inverse() * (vt0 - vs0);
			idxvt0n = meshS0.cfv_begin(*itFaceS0)->idx();
			firstLoop = false;
		}

		// set b
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				b[j][rowidx + i] = Vtn(j, i);

		// set A
		auto it = meshS0.cfv_begin(*itFaceS0);
		int idxV1 = it->idx();
		int idxV2 = (++it)->idx();
		int idxV3 = (++it)->idx();

		*pja++ = idxV2;					
		*pja++ = idxV1;					
		*pja++ = idxV3;					
		*pja++ = idxV1;					
		*pja++ = idxV1 + verticeNum;	// index for v4 is v1 + |v|
		*pja++ = idxV1;	

		// a[] = { 1 -1 1 -1 ... 1 }, so a[] is set after the loop
		// ia[] = { 0 2 4 6 8 ... 2n 2n+1 }，so ia[] is set after the loop

		rowidx += 3;
	}

	// set a[] and ia[]
	int e = -1, f = -2;
	generate_n(a, nznum, [&e]() { return (e = -e); });
	generate_n(ia, rownum + 1, [&f]() { return (f += 2); });
	--ia[rownum];		// the last row contains only one element

	// use vt0n to set global offset
	*pja = idxvt0n;
	b[0][rowidx] = vt0n(0);
	b[1][rowidx] = vt0n(1);
	b[2][rowidx] = vt0n(2);

	/* To avoid constantly repeating the part of code that checks inbound SparseBLAS functions' status,
    use macro CALL_AND_CHECK_STATUS

    SPARSE_STATUS_SUCCESS           = 0,    the operation was successful
    SPARSE_STATUS_NOT_INITIALIZED   = 1,    empty handle or matrix arrays
    SPARSE_STATUS_ALLOC_FAILED      = 2,    internal error: memory allocation failed
    SPARSE_STATUS_INVALID_VALUE     = 3,    invalid input value
    SPARSE_STATUS_EXECUTION_FAILED  = 4,    e.g. 0-diagonal element for triangular solver, etc.
    SPARSE_STATUS_INTERNAL_ERROR    = 5,    internal error
    SPARSE_STATUS_NOT_SUPPORTED     = 6     e.g. operation for double precision doesn't support other types
	*/

#define CALL_AND_CHECK_STATUS(function, error_message) do { \
          status = function;                                \
          if(status != SPARSE_STATUS_SUCCESS) {             \
          printf(error_message); fflush(0);                 \
          printf("exit status is %d\n", status); fflush(0); \
		  exit(status);			                            \
          }                                                 \
} while(0)

	// allocate ATb and x
	double *const ATbb = new double[colnum * 3];
	double *const xb = new double[colnum * 3];
	double *const ATb[3] = { ATbb, ATbb + colnum, ATbb + 2 * colnum };
	double *const x[3] = { xb, xb + colnum, xb + 2 * colnum };

	// create A
	CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, rownum, colnum, ia, ia + 1, ja, a),
						  "Error after MKL_SPARSE_D_CREATE_CSR, csrA \n");

	// calculate ATA
	sparse_matrix_t csrATA;
	CALL_AND_CHECK_STATUS(mkl_sparse_order(csrA),
						  "Error after MKL_SPARSE_ORDER\n");

	CALL_AND_CHECK_STATUS(mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, csrA, &csrATA),
						  "Error after MKL_SPARSE_SYRK\n");

	// calculate ATb
	CALL_AND_CHECK_STATUS(mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, 1.0, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR,
										  bb, 3, rownum, 0.0, ATbb, colnum),
						  "Error after MKL_SPARSE_D_MM\n");

	// exprot ATA csr
	int inotuse;
	double dnotuse, *a2;
	MKL_INT *pnotuse, *ia2, *ja2;
	sparse_index_base_t snotuse;
	CALL_AND_CHECK_STATUS(mkl_sparse_d_export_csr(csrATA, &snotuse, &inotuse, &inotuse, &ia2,
												  &pnotuse, &ja2, &a2),
						  "Error after MKL_SPARSE_D_EXPORT_CSR\n");			// ia2 == pnotuse - 1
	
	// setup pardiso control parameters
	MKL_INT phase;			// set later
	MKL_INT mtype = -2;		// real and symmetric 
	MKL_INT nrhs = 3;		// 3 right-hand sides
	MKL_INT n = colnum;		// number of equations in the sparse linear systems
	MKL_INT maxfct = 1;		// Maximum number of numerical factorizations
	MKL_INT mnum = 1;		// Which factorization to use
	MKL_INT msglvl = 0;		// DONOT print statistical information in file 
	MKL_INT error = 0;      // Initialize error flag 
	void *pt[64] = {};		// internal solver memory pointer

	MKL_INT iparm[64] = {};	// pardiso control parameters
	iparm[0] = 1;		    // No solver default 
	iparm[1] = 3;     	    // Openmp algorithm 
	iparm[3] = 0;     	    // No iterative-direct algorithm 
	iparm[4] = 0;     	    // No user fill-in reducing permutation 
	iparm[5] = 0;     	    // Write solution into x 
	iparm[7] = 0;     	    // Max numbers of iterative refinement steps 
	iparm[9] = 2;     	    // Perturb the pivot elements with 1E-2 
	iparm[10] = 1;    	    // Use nonsymmetric permutation and scaling MPS 
	iparm[12] = 0;    	    // Maximum weighted matching algorithm is switched-off
	iparm[13] = 0;    	    // Output: Number of perturbed pivots 
	iparm[17] = -1;   	    // Output: Number of nonzeros in the factor LU 
	iparm[18] = -1;   	    // Output: Mflops for LU factorization 
	iparm[19] = 0;    	    // Output: Numbers of CG Iterations 
	iparm[34] = 1;    	    // PARDISO use C-style indexing for ia and ja arrays 
	
	// Reordering and Symbolic Factorization. This step also allocates 
	// all memory that is necessary for the factorization.
	phase = 11;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
			&n, a2, ia2, ja2, &inotuse, &nrhs, iparm, &msglvl, &dnotuse, &dnotuse, &error);
	if (error != 0) {
		printf("\nERROR during symbolic factorization: %d", error);
		exit(1);
	}

	// Numerical factorization
 	phase = 22;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
			&n, a2, ia2, ja2, &inotuse, &nrhs, iparm, &msglvl, &dnotuse, &dnotuse, &error);
	if (error != 0) {
		printf("\nERROR during numerical factorization: %d", error);
		exit(2);
	}
	
	// Back substitution and iterative refinement
	phase = 33;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
			&n, a2, ia2, ja2, &inotuse, &nrhs, iparm, &msglvl, ATbb, xb, &error);
	if (error != 0) {
		printf("\nERROR during solution: %d", error);
		exit(3);
	}
	
	// Termination and release of memory
 	phase = -1; 
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
			&n, &dnotuse, ia2, ja2, &inotuse, &nrhs,
			iparm, &msglvl, &dnotuse, &dnotuse, &error);


	// write obj model
	OpenMesh::VertexHandle *hVerticeArray = new OpenMesh::VertexHandle[verticeNum];

	for (int i = 0; i < verticeNum; i++) {	// add vertice
		hVerticeArray[i] = meshT1.add_vertex(OpenMesh::Vec3f(x[0][i], x[1][i], x[2][i]));
	}

	for (auto itFaceT0 = meshT0.faces_begin(); itFaceT0 != meshT0.faces_end(); ++itFaceT0) {	// add faces
		auto it = meshT0.cfv_begin(*itFaceT0);
		int idxV1 = it->idx();
		int idxV2 = (++it)->idx();
		int idxV3 = (++it)->idx();
		meshT1.add_face(hVerticeArray[idxV1], hVerticeArray[idxV2], hVerticeArray[idxV3]);
	}

	if (!OpenMesh::IO::write_mesh(meshT1, t1_obj_path)) exit(1);

	// deallocate	
	if (mkl_sparse_destroy(csrA) != SPARSE_STATUS_SUCCESS) {
		printf("Error after MKL_SPARSE_DESTROY, csrA \n"); fflush(0); exit(1);
	}

	if (mkl_sparse_destroy(csrATA) != SPARSE_STATUS_SUCCESS) {
		printf(" Error after MKL_SPARSE_DESTROY, csrATA \n"); fflush(0); exit(1);
	}

	delete[] bb;
	delete[] ia;
	delete[] ja;
	delete[] a;
	delete[] ATbb;
	delete[] xb;
	delete[] hVerticeArray;
}


int main(int argc, char *argv[]) {
	if (argc == 1)  deformation_transfer("s0.obj", "s1.obj", "t0.obj", "t1.obj");
	else if (argc == 5) deformation_transfer(argv[1], argv[2], argv[3], argv[4]);
	else 
		cout << "Usage: " << argv[0]
			 << " [<s0_obj_name> <s1_obj_name> <t0_obj_name> <t1_obj_name>]" << endl;
	return 0;
}

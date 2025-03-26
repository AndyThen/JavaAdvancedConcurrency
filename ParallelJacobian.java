// All of this is just from Utilities, but I'm leaving this here to show all modules needed.
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Parallelized Jacobian.
 */
public class ParallelJacobian {
    
    // Small value for approximating derivatives
    private static final double EPSILON = 1e-6;
    
    // Matrix dimensions
    private static final int MATRIX_SIZE = 100000;
    
    // Fetchces the number of processors available on your machine. Check JavaDoc for more info.
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    
    // This is a simplified sparse representation using row-column-value triplets
    private static class SparseMatrix {
        private int[] rows;
        private int[] cols;
        private double[] values;
        private int size;
        private int maxSize;
        private int dim;
        
        public SparseMatrix(int dimension, int maxElements) {
            this.dim = dimension;
            this.maxSize = maxElements;
            this.rows = new int[maxElements];
            this.cols = new int[maxElements];
            this.values = new double[maxElements];
            this.size = 0;
        }
        
        public synchronized void set(int row, int col, double value) {
            if (size < maxSize) {
                rows[size] = row;
                cols[size] = col;
                values[size] = value;
                size++;
            }
        }
        
        public double get(int row, int col) {
            for (int i = 0; i < size; i++) {
                if (rows[i] == row && cols[i] == col) {
                    return values[i];
                }
            }
            return 0.0;
        }
    }
    
    // Cached matrices
    private static SparseMatrix matrixA;
    private static SparseMatrix matrixB;
    
    /**
     * Initialize the large sparse matrices used in the computation in parallel
     */
    private static void initializeMatrices() {
        System.out.println("Initializing sparse " + MATRIX_SIZE + "×" + MATRIX_SIZE + " matrices with " + NUM_THREADS + " threads...");
        long startInit = System.currentTimeMillis();
        
        Random random = new Random(42); 
        
        // Create sparse matrices with a limited number of non-zero entries
        // Using 0.001% density for extremely large matrices
        int nonZeroEntries = (int)(MATRIX_SIZE * MATRIX_SIZE * 0.00001);
        System.out.println("Creating sparse matrices with " + nonZeroEntries + " non-zero entries each");
        
        matrixA = new SparseMatrix(MATRIX_SIZE, nonZeroEntries);
        matrixB = new SparseMatrix(MATRIX_SIZE, nonZeroEntries);
        
        // Parallelize matrix initialization
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        List<Future<?>> futures = new ArrayList<>();
        
        // Calculate entries per thread
        int entriesPerThread = nonZeroEntries / NUM_THREADS;
        
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            final int startEntry = t * entriesPerThread;
            final int endEntry = (t == NUM_THREADS - 1) ? nonZeroEntries : (t + 1) * entriesPerThread;
            
            futures.add(executor.submit(() -> {
                Random threadRandom = new Random(42 + threadId); // Different seed per thread, but still deterministic
                
                for (int i = startEntry; i < endEntry; i++) {
                    int rowA = threadRandom.nextInt(MATRIX_SIZE);
                    int colA = threadRandom.nextInt(MATRIX_SIZE);
                    double valueA = threadRandom.nextDouble() * 0.1;
                    matrixA.set(rowA, colA, valueA);
                    
                    int rowB = threadRandom.nextInt(MATRIX_SIZE);
                    int colB = threadRandom.nextInt(MATRIX_SIZE);
                    double valueB = threadRandom.nextDouble() * 0.1;
                    matrixB.set(rowB, colB, valueB);
                    
                    // Print progress every 10% completion to stdout
                    if ((i - startEntry) % (entriesPerThread / 10) == 0) {
                        System.out.println("Thread " + threadId + " matrix initialization " + 
                                          ((i - startEntry) * 100 / entriesPerThread) + "% complete");
                    }
                }
                return null;
            }));
        }
        
        // Check for completed thread, fixed a couple of things.
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        executor.shutdown();
        
        long endInit = System.currentTimeMillis();
        System.out.println("Matrix initialization completed in " + 
                          (endInit - startInit) / 1000.0 + " seconds");
    }
    
    /**
     * Matrix multiplication function that operates on a subregion of the matrices
     * based on input parameters to make the result dependent on the input vector.
     * Parallel execution for the matrix multiplication
     * 
     * @param x Input vector that influences the matrix multiplication
     * @return A vector containing results derived from the matrix multiplication
     */
    public static double[] computeIntensiveFunction(double[] x) {
        int dimension = x.length;
        double[] result = new double[dimension];
        
        // Create matrix multiplication region deterministically from input parameters
        int startRowA = (int)(Math.abs(x[0]) * 1000) % (MATRIX_SIZE - 1000);
        int startColA = (int)(Math.abs(x[1 % dimension]) * 1000) % (MATRIX_SIZE - 1000);
        int startRowB = startColA;  // For matrix multiplication compatibility
        int startColB = (int)(Math.abs(x[2 % dimension]) * 1000) % (MATRIX_SIZE - 1000);
        
        // Size of submatrix to multiply. Using a small submatrix size for less process granularity
        int subSize = 100 + (int)(Math.abs(x[3 % dimension]) * 50) % 200;
        
        // Bounds checking
        if (startRowA + subSize > MATRIX_SIZE) startRowA = MATRIX_SIZE - subSize;
        if (startColA + subSize > MATRIX_SIZE) startColA = MATRIX_SIZE - subSize;
        if (startRowB + subSize > MATRIX_SIZE) startRowB = MATRIX_SIZE - subSize;
        if (startColB + subSize > MATRIX_SIZE) startColB = MATRIX_SIZE - subSize;
        
        System.out.println("Computing multiplication for submatrix at positions: (" +
                          startRowA + "," + startColA + ") and (" + startRowB + "," + startColB + 
                          ") with size " + subSize + " using " + NUM_THREADS + " threads");
        
        // Prepare result matrix
        double[][] subResult = new double[subSize][subSize];
        
        // Parallelize matrix multiplication using an ExecutorService. Check JavaDoc for more info.
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        List<Future<?>> futures = new ArrayList<>();
        
        // Calculate rows per thread
        int rowsPerThread = subSize / NUM_THREADS;
        
        final int finalStartRowA = startRowA;
        final int finalStartColA = startColA;
        final int finalStartRowB = startRowB;
        final int finalStartColB = startColB;
        
        // Assign rows to different threads
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            final int startRow = t * rowsPerThread;
            final int endRow = (t == NUM_THREADS - 1) ? subSize : (t + 1) * rowsPerThread;
            
            futures.add(executor.submit(() -> {
                // Process assigned rows
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < subSize; j++) {
                        subResult[i][j] = 0;
                        for (int k = 0; k < subSize; k++) {
                            subResult[i][j] += matrixA.get(finalStartRowA + i, finalStartColA + k) * 
                                             matrixB.get(finalStartRowB + k, finalStartColB + j);
                        }
                        // Scale result by input parameter to make it sensitive to all inputs
                        subResult[i][j] *= (1 + Math.sin(x[i % dimension]));
                    }
                    
                    // Print progress every 10% completion to stdout
                    if ((i - startRow) % (rowsPerThread / 10) == 0) {
                        System.out.println("Thread " + threadId + " matrix multiplication " + 
                                          ((i - startRow) * 100 / rowsPerThread) + "% complete");
                    }
                }
                return null;
            }));
        }
        
        // Wait for all threads to complete
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        executor.shutdown();
        
        // Extract results for our output vector
        for (int i = 0; i < dimension; i++) {
            // Take values from different parts of the result matrix
            int row = (i * 13) % subSize;
            int col = (i * 17) % subSize;
            result[i] = subResult[row][col];
            
            // Add some transcendental functions to make gradients more interesting
            result[i] += Math.sin(x[i]) + Math.cos(x[(i+1) % dimension]);
        }
        
        return result;
    }
    
    /**
     * Computes the Jacobian matrix of a vector-valued function at a specific point.
     * 
     * @param function The vector-valued function for which to compute the Jacobian
     * @param point The point at which to compute the Jacobian
     * @param functionDimension The dimension of the output vector of the function
     * @return The Jacobian matrix
     */
    public static double[][] computeJacobian(Function<double[], double[]> function, 
                                            double[] point, 
                                            int functionDimension) {
        int variableCount = point.length;
        double[][] jacobian = new double[functionDimension][variableCount];
        
        // Evaluation of base function
        long startBaseEval = System.currentTimeMillis();
        System.out.println("Computing base function evaluation...");
        double[] baseFunctionValue = function.apply(point);
        long endBaseEval = System.currentTimeMillis();
        
        System.out.println("Base function evaluated in " + 
                          (endBaseEval - startBaseEval) / 1000.0 + " seconds");
        System.out.println("Base function values: " + Arrays.toString(baseFunctionValue));
        
        // Parallelize computation of partial derivatives
        System.out.println("Computing partial derivatives in parallel using " + NUM_THREADS + " threads");
        long startPartials = System.currentTimeMillis();
        
        // https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/CompletableFuture.html
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        for (int j = 0; j < variableCount; j++) {
            final int varIndex = j;
            
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                System.out.println("Thread computing partial derivatives for variable " + varIndex);
                long startPartial = System.currentTimeMillis();
                
                // Create perturbed point
                double[] perturbedPoint = Arrays.copyOf(point, point.length);
                perturbedPoint[varIndex] += EPSILON;
                
                // Evaluate function at perturbed point
                double[] perturbedValue = function.apply(perturbedPoint);
                
                // Computation of partial derivative
                for (int i = 0; i < functionDimension; i++) {
                    // Forward difference approximation, Reference: https://www.sheffield.ac.uk/media/32080/download?attachment
                    jacobian[i][varIndex] = (perturbedValue[i] - baseFunctionValue[i]) / EPSILON;
                }
                
                long endPartial = System.currentTimeMillis();
                System.out.println("  - Variable " + varIndex + " completed in " + 
                                  (endPartial - startPartial) / 1000.0 + " seconds");
            });
            
            futures.add(future);
        }
        
        // https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/CompletableFuture.html
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(
            futures.toArray(new CompletableFuture[0])
        );
        
        // Check later
        try {
            allFutures.join();
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        long endPartials = System.currentTimeMillis();
        System.out.println("All partial derivatives computed in " + 
                          (endPartials - startPartials) / 1000.0 + " seconds");
        
        return jacobian;
    }
    
    /**
     * Prints a matrix to stdout
     * 
     * @param matrix The matrix to print
     * @param name The name of the matrix to display
     */
    public static void printMatrix(double[][] matrix, String name) {
        System.out.println(name + ":");
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.printf("%10.6f ", matrix[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Starting PARALLELIZED Jacobian calculation...");
        System.out.println("Using function involving " + MATRIX_SIZE + "×" + MATRIX_SIZE + " matrix multiplication");
        System.out.println("Utilizing " + NUM_THREADS + " threads (detected " + 
                          Runtime.getRuntime().availableProcessors() + " available processors)");
        
        long startTime = System.currentTimeMillis();
        
        // Init matrices of larger size
        initializeMatrices();
        
        // Create a point for evaluation
        int dimension = 3; // Small dimension, but function is extremely intensive
        double[] point = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            point[i] = 0.5 + i * 0.1; // Init with different values
        }
        
        System.out.println("Evaluating at point: " + Arrays.toString(point));
        System.out.println("Output dimension: " + dimension);
        
        // Wrap our intensive function in a Function interface
        Function<double[], double[]> function = ParallelJacobian::computeIntensiveFunction;
        
        // Compute Jacobian matrix
        System.out.println("Computing Jacobian in parallel: ");
        double[][] jacobian = computeJacobian(function, point, dimension);
        
        printMatrix(jacobian, "Jacobian Matrix");
        
        long endTime = System.currentTimeMillis();
        System.out.println("Computation completed in " + (endTime - startTime) / 1000.0 + " seconds");
        
        double sum = 0;
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                sum += Math.abs(jacobian[i][j]);
                max = Math.max(max, jacobian[i][j]);
                min = Math.min(min, jacobian[i][j]);
            }
        }
        
        System.out.println("Jacobian statistics:");
        System.out.println("Average absolute value: " + (sum / (dimension * dimension)));
        System.out.println("Maximum value: " + max);
        System.out.println("Minimum value: " + min);
    }
}

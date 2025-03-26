import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 * Sequential Jacobian of a function involving 100,000 × 100,000 matrix multiplications.
 * 
 */
public class MatrixMultiplicationJacobi {
    
    // Small value for approximating derivatives
    private static final double EPSILON = 1e-6;
    
    // Matrix dimensions
    private static final int MATRIX_SIZE = 100000;
    
    // This is a simplified sparse representation using row-column-value triplets
    private static class SparseMatrix {
        // Using sparse representation to save memory
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
        
        public void set(int row, int col, double value) {
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
     * Initialize the large sparse matrices used in the computation
     */
    private static void initializeMatrices() {
        System.out.println("Initializing sparse " + MATRIX_SIZE + "×" + MATRIX_SIZE + " matrices...");
        long startInit = System.currentTimeMillis();
        
        Random random = new Random(42); 
        
        // Create sparse matrices with a limited number of non-zero entries
        // Using 0.001% density for extremely large matrices
        int nonZeroEntries = (int)(MATRIX_SIZE * MATRIX_SIZE * 0.00001);
        System.out.println("Creating sparse matrices with " + nonZeroEntries + " non-zero entries each");
        
        matrixA = new SparseMatrix(MATRIX_SIZE, nonZeroEntries);
        matrixB = new SparseMatrix(MATRIX_SIZE, nonZeroEntries);
        
        // Fill with random non-zero values
        for (int i = 0; i < nonZeroEntries; i++) {
            int rowA = random.nextInt(MATRIX_SIZE);
            int colA = random.nextInt(MATRIX_SIZE);
            double valueA = random.nextDouble() * 0.1;
            matrixA.set(rowA, colA, valueA);
            
            int rowB = random.nextInt(MATRIX_SIZE);
            int colB = random.nextInt(MATRIX_SIZE);
            double valueB = random.nextDouble() * 0.1;
            matrixB.set(rowB, colB, valueB);
            
            // Print progress every 10% of initialization
            if (i % (nonZeroEntries / 10) == 0) {
                System.out.println("Matrix initialization " + (i * 100 / nonZeroEntries) + "% complete");
            }
        }
        
        long endInit = System.currentTimeMillis();
        System.out.println("Matrix initialization completed in " + 
                          (endInit - startInit) / 1000.0 + " seconds");
    }
    
    /**
     * Matrix multiplication function that operates on a subregion of the matrices
     * based on input parameters to make the result dependent on the input vector.
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
        
        // Size of submatrix to multiply. Using a relatively small submatrix size to make the computation feasible
        int subSize = 100 + (int)(Math.abs(x[3 % dimension]) * 50) % 200;
        
        // Bounds checking
        if (startRowA + subSize > MATRIX_SIZE) startRowA = MATRIX_SIZE - subSize;
        if (startColA + subSize > MATRIX_SIZE) startColA = MATRIX_SIZE - subSize;
        if (startRowB + subSize > MATRIX_SIZE) startRowB = MATRIX_SIZE - subSize;
        if (startColB + subSize > MATRIX_SIZE) startColB = MATRIX_SIZE - subSize;
        
        System.out.println("Computing multiplication for submatrix at positions: (" +
                          startRowA + "," + startColA + ") and (" + startRowB + "," + startColB + 
                          ") with size " + subSize);
        
        // Perform matrix multiplication on subregion
        double[][] subResult = new double[subSize][subSize];
        
        for (int i = 0; i < subSize; i++) {
            for (int j = 0; j < subSize; j++) {
                subResult[i][j] = 0;
                for (int k = 0; k < subSize; k++) {
                    subResult[i][j] += matrixA.get(startRowA + i, startColA + k) * 
                                      matrixB.get(startRowB + k, startColB + j);
                }
                // Scale result by input parameter to make it sensitive to all inputs
                subResult[i][j] *= (1 + Math.sin(x[i % dimension]));
            }
            
            // Print progress for the multiplication
            if (i % (subSize / 10) == 0) {
                System.out.println("Matrix multiplication " + (i * 100 / subSize) + "% complete");
            }
        }
        
        // Extract results for our output vector
        for (int i = 0; i < dimension; i++) {
            // Take values from different parts of the result matrix
            int row = (i * 13) % subSize;
            int col = (i * 17) % subSize;
            result[i] = subResult[row][col];
            
            // Transcendental functions 
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
        
        // Base function evaluation at the point
        long startBaseEval = System.currentTimeMillis();
        System.out.println("Computing base function evaluation...");
        double[] baseFunctionValue = function.apply(point);
        long endBaseEval = System.currentTimeMillis();
        
        System.out.println("Base function evaluated in " + 
                          (endBaseEval - startBaseEval) / 1000.0 + " seconds");
        System.out.println("Base function values: " + Arrays.toString(baseFunctionValue));
        
        // For each variable, compute partial derivatives for all function components
        for (int j = 0; j < variableCount; j++) {
            System.out.println("Computing partial derivatives for variable " + j + 
                              " (" + (j+1) + " of " + variableCount + ")");
            long startPartial = System.currentTimeMillis();
            
            // Create perturbed point
            double[] perturbedPoint = Arrays.copyOf(point, point.length);
            perturbedPoint[j] += EPSILON;
            
            // Evaluate function at perturbed point
            double[] perturbedValue = function.apply(perturbedPoint);
            
            // Compute partial derivatives for this variable
            for (int i = 0; i < functionDimension; i++) {
                // Forward difference approximation, Reference: https://www.sheffield.ac.uk/media/32080/download?attachment
                jacobian[i][j] = (perturbedValue[i] - baseFunctionValue[i]) / EPSILON;
            }
            
            long endPartial = System.currentTimeMillis();
            System.out.println("  - Completed in " + (endPartial - startPartial) / 1000.0 + " seconds");
        }
        
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
        System.out.println("Starting...");
        System.out.println("Using function involving " + MATRIX_SIZE + "×" + MATRIX_SIZE + " matrix multiplication");
      
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
        Function<double[], double[]> function = MatrixMultiplicationJacobi::computeIntensiveFunction;
        
        // Compute the Jacobian matrix
        System.out.println("Computing Jacobian - this will take significant time...");
        double[][] jacobian = computeJacobian(function, point, dimension);
        
        // Print results
        printMatrix(jacobian, "Jacobian Matrix");
        
        long endTime = System.currentTimeMillis();
        System.out.println("Computation completed in " + (endTime - startTime) / 1000.0 + " seconds");
        
        // Calculate and display some statistics
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

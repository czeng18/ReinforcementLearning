package RL;

import java.util.Arrays;

/**
 * Operations on matrices
 *
 * @author Caroline Zeng
 * @version 1.0.0
 */ 

public class Utility {
    /*
     * height = x = mat.length    = # of rows    = column length
     * width  = y = mat[n].length = # of columns = row length
     *
     * mat[y] = row
     */

    /**
     * Multiply 2 matrices, given their dimensions allow them to be multiplied together
     * @param mat1  left hand AxB matrix
     * @param mat2  right hand BxC matrix
     * @return      AxC matrix
     */
    public static double[][] matrixMultiply(double[][] mat1, double[][] mat2)
    {
        if (mat1[0].length != mat2.length)
        {
            System.out.println("UNMULTIPLIABLE");
            return null;
        }
        int retx      = getX(mat2);
        int rety      = getY(mat1);

        double[][] ret = new double[rety][retx];

        for (int x = 0; x < retx; x++)
        {

            for (int y = 0; y < rety; y++)
            {

                double[] row = getRow(mat1, y);
                double[] col = getCol(mat2, x);
                double[] res = new double[row.length];

                for (int i = 0; i < row.length; i++)
                {
                    res[i] = row[i] * col[i];
                }

                ret[y][x] = sumOfAll(res);

            }

        }

        return ret;
    }

    /**
     * Multiplies a matrix by a scalar
     * @param mat   matrix to be multiplied
     * @param s     scalar to multiply matrix by
     * @return      matrix multiplied by scalar
     */
    public static double[][] scalarMultiply(double[][] mat, double s)
    {
        int x         = getX(mat);
        int y         = getY(mat);
        double[][] ret = new double[y][x];

        for (int i = 0; i < y; i++)
        {

            for (int j = 0; j < x; j++)
            {
                ret[i][j] = mat[i][j] * s;
            }

        }

        return ret;
    }

    /**
     * Gets the matrix of minors for a matrix
     * @param mat   matrix to find matrix of minors for
     * @return      matrix of minors
     */
    public static double[][] getMatMinors(double[][] mat)
    {
        int dim       = mat.length;
        double[][] ret = new double[dim][dim];

        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                boolean passx   = false;
                double[][] small = new double[dim - 1][dim - 1];

                for (int i = 0; i < dim; i++)
                {
                    boolean passy = false;

                    if (i == x)
                    {
                        passx = true;
                        continue;
                    }

                    int indx = i;
                    if (passx) indx--;

                    for (int j = 0; j < dim; j++)
                    {
                        if (j == y)
                        {
                            passy = true;
                            continue;
                        }

                        int indy = j;
                        if (passy) indy--;
                        small[indx][indy] = mat[i][j];
                    }
                }
                ret[x][y] = getDet(small);
            }
        }
        return ret;
    }

    /**
     * Gets the matrix of cofactors for a matrix
     * @param mat   matrix to find matrix of cofactors for
     * @return      matrix of cofactors
     */
    public static double[][] getMatCofactors(double[][] mat)
    {
        int dim          = mat.length;
        double[][] ret    = new double[dim][dim];
        double[][] matmin = getMatMinors(mat);

        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                double val = matmin[x][y];
                if ((x + y) % 2 == 1) val = -val;
                ret[x][y] = val;
            }
        }
        ret = transpose(ret);
        return ret;
    }

    /**
     * Gets the transpose of a matrix
     * Essentially flips matrix along its diagonal
     * @param mat   matrix to find transpose for
     * @return      transpose of matrix
     */
    public static double[][] transpose(double[][] mat)
    {
        int x         = mat.length;
        int y         = mat[0].length;
        double[][] ret = new double[y][x];

        for (int i = 0; i < y; i++)
        {
            for (int j = 0; j < x; j++)
            {
                ret[i][j] = mat[j][i];
            }
        }
        return ret;
    }

    /**
     * Gets the transpose of a matrix
     * Essentially flips matrix along its diagonal
     * @param mat   matrix to find transpose for
     * @return      transpose of matrix
     */
    public static double[][] transpose (double[] mat)
    {
        int x = 1;
        int y = mat.length;
        double[][] ret = new double[y][x];
        for (int i = 0; i < y; i++)
        {
            ret[i][0] = mat[i];
        }
        return ret;
    }

    /**
     * Gets the inverse of a matrix
     * @param mat   matrix to find inverse for
     * @return      inverse of matrix
     */
    public static double[][] getInverse(double[][] mat)
    {
        double det   = getDet(mat);
        double[][] C = getMatCofactors(mat);
        return scalarMultiply(C, (1/det));
    }

    /**
     * Gets determinant of matrix
     * @param mat   matrix to find determinant of
     * @return      determinant of matrix
     */
    public static double getDet(double[][] mat)
    {
        double ret = 0;
        int dim   = mat.length;

        if (dim > 2)
        {
            double[] toprow = getRow(mat, 0);

            for (int i = 0; i < dim; i++)
            {
                double t         = toprow[i];
                double[][] small = new double[dim - 1][dim - 1];
                boolean passed  = false;

                for (int j = 0; j < dim; j++)
                {
                    if (j == i)
                    {
                        passed = true;
                        continue;
                    }

                    int index = j;
                    if (passed) index--;
                    small = setCol(small, Arrays.copyOfRange(getCol(mat, j), 1, dim), index);
                }
                if (i % 2 == 0) ret += t * getDet(small);
                else ret -= t * getDet(small);
            }
        } else
        {
            ret = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
        }
        return ret;
    }

    /**
     * Gets a row of a matrix
     * @param mat   matrix to get row from
     * @param x     y-value of the row
     * @return      row of matrix
     */
    public static double[] getCol(double[][] mat, int x)
    {
        double[] ret = new double[mat.length];

        for (int i = 0; i < mat.length; i++)
        {
            double[] col = mat[i];
            ret[i]       = col[x];
        }
        return ret;
    }

    /**
     * Gets a column of a matrix
     * @param mat   matrix to get column from
     * @param y     x-value of column
     * @return      column of matrix
     */
    public static double[] getRow(double[][] mat, int y)
    {
        double[] ret = mat[y];
        return ret;
    }

    /**
     * Sets a row of a matrix to given values
     * @param mat   matrix to set row
     * @param row   row to set matrix's row to
     * @param x     x-value of row in matrix
     * @return      matrix with new values in given row
     */
    public static double[][] setCol(double[][] mat, double[] row, int x)
    {
        for (int i = 0; i < mat.length; i++)
        {
            mat[i][x] = row[i];
        }
        return mat;
    }

    /**
     * Sets a column of a matrix to given values
     * @param mat   matrix to set column
     * @param col   column to set matrix's column to
     * @param y     y-value of column in matrix
     * @return      matrix with new values in given column
     */
    public static double[][] setRow(double[][] mat, double[] col, int y)
    {
        mat[y] = col;
        return mat;
    }

    /**
     * Prints a row of a matrix
     * @param mat   matrix to print row of
     * @param y     y-value of row
     */
    public static void printRow(double[][] mat, int y)
    {
        double[] row = getRow(mat, y);
        String out  = "";

        for (double i : row)
        {
            out += i + " ";
        }
        System.out.println(out);
    }

    /**
     * Prints a matrix
     * @param mat   matrix to print
     */
    public static void printMat(double[][] mat)
    {
        for (int i = 0; i < getY(mat); i++)
        {
            printRow(mat, i);
        }
    }

    /**
     * Gets x-dimension of a matrix
     * @param mat   matrix to find x-dimension of
     * @return      x-dimension of matrix
     */
    public static int getY(double[][] mat)
    {
        return mat.length;
    }

    /**
     * Gets y-dimension of matrix
     * @param mat   matrix to get y-dimension of
     * @return      y-dimension of matrix
     */
    public static int getX(double[][] mat)
    {
        return mat[0].length;
    }

    /**
     * Performs sigmoid function on an individual value
     * @param x value
     * @return  transformed "squashed" value
     */
    public static double sigmoidInd(double x)
    {
        return (double) (1 / (1 + Math.pow(Math.E, -x)));
    }

    /**
     * Performs sigmoid prime function on an individual value
     * @param x value
     * @return  transformed value
     */
    public static double sigmoidPrimInd(double x)
    {
        // d/dx((1 + e^-x)^-1) = -1 * (1 + e^-x)^-2 * e^-x * -1 = (1 + e^-x)^-2 * e^-x
        return (double)sigmoidInd(x) * (1 - sigmoidInd(x));
    }

    /**
     * Performs sigmoid function on all values in a matrix
     * @param mat   matrix
     * @return      matrix of "squashed" values
     */
    public static double[][] sigmoidMat(double[][] mat)
    {
        for (int x = 0; x < getX(mat); x++)
        {
            for (int y = 0; y < getY(mat); y++)
            {
                mat[x][y] = sigmoidInd(mat[x][y]);
            }
        }
        return mat;
    }

    public static double[][] sigmoidPrimeMat(double[][] mat)
    {
        for (int x = 0; x < getX(mat); x++)
        {
            for (int y = 0; y < getY(mat); y++)
            {
                mat[y][x] = sigmoidPrimInd(mat[y][x]);
            }
        }
        return mat;
    }

    public static double[][] scalarSubMat(double[][] mat1, double[][] mat2)
    {
        if (getX(mat1) != getX(mat2) || getY(mat1) != getY(mat2)) return null;
        double[][] result = new double[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++)
        {
            for (int j = 0; j < mat1[0].length; j++)
            {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        return result;
    }

    public static double[][] scalarAddMat(double[][] mat1, double[][] mat2)
    {
        if (getX(mat1) != getX(mat2) || getY(mat1) != getY(mat2)) return null;
        double[][] result = new double[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++)
        {
            for (int j = 0; j < mat1[0].length; j++)
            {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }

    public static double[][] scalarMultMat(double[][] mat1, double[][] mat2)
    {
        if (getX(mat1) != getX(mat2) || getY(mat1) != getY(mat2)) return null;
        double[][] result = new double[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++)
        {
            for (int j = 0; j < mat1[0].length; j++)
            {
                result[i][j] = mat1[i][j] * mat2[i][j];
            }
        }
        return result;
    }

    public static double sumOfAll(double[] mat)
    {
        double sum = 0;
        for (double i : mat)
        {
            sum += i;
        }
        return sum;
    }

    public static double sumOfAll(double[][] mat)
    {
        double res = 0;
        for (double[] m : mat)
        {
            for (double entry : m)
            {
                res += entry;
            }
        }
        return res;
    }

    public static double[] scalarMultiply(double[] mat, double scalar)
    {
        for (int i = 0; i < mat.length; i++)
        {
            mat[i] = scalar * mat[i];
        }
        return mat;
    }

    public static boolean areSameSize(double[][] mat1, double[][] mat2)
    {
        if (mat1.length == mat2.length && mat1[0].length == mat2[0].length) return true;
        return false;
    }

    public static boolean areMultipliable(double[][] mat1, double[][] mat2)
    {
        if (mat1[0].length == mat2.length) return true;
        return false;
    }
}
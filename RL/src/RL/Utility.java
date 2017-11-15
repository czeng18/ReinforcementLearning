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
     * width  = x = mat.length    = row
     * height = y = mat[n].length = column
     *
     * plaintext being ciphered = float[0][textlength]
     *
     * multiplication result = float[key.x][text.y]
     *
     * get from matrix = mat[x][y]
     */

    /**
     * Multiply 2 matrices, given their dimensions allow them to be multiplied together
     * @param mat1  left hand AxB matrix
     * @param mat2  right hand BxC matrix
     * @return      AxC matrix
     */
    public static float[][] matrixMultiply(float[][] mat1, float[][] mat2)
    {
        int retx      = getX(mat2);
        int rety      = getY(mat1);
        float[][] ret = new float[retx][rety];

        for (int x = 0; x < retx; x++)
        {

            for (int y = 0; y < rety; y++)
            {

                float sum     = 0;
                float[] row = getRow(mat1, y);
                float[] col = getCol(mat2, x);

                for (int i = 0; i < row.length; i++)
                {
                    sum += row[i] * col[i];
                }

                ret[x][y] = sum;

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
    public static float[][] scalarMultiply(float[][] mat, float s)
    {
        int x         = getX(mat);
        int y         = getY(mat);
        float[][] ret = new float[x][y];

        for (int i = 0; i < x; i++)
        {

            for (int j = 0; j < y; j++)
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
    public static float[][] getMatMinors(float[][] mat)
    {
        int dim       = mat.length;
        float[][] ret = new float[dim][dim];

        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                boolean passx   = false;
                float[][] small = new float[dim - 1][dim - 1];

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
    public static float[][] getMatCofactors(float[][] mat)
    {
        int dim          = mat.length;
        float[][] ret    = new float[dim][dim];
        float[][] matmin = getMatMinors(mat);

        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                float val = matmin[x][y];
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
    public static float[][] transpose(float[][] mat)
    {
        int x         = mat.length;
        int y         = mat[0].length;
        float[][] ret = new float[y][x];

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
     * Gets the inverse of a matrix
     * @param mat   matrix to find inverse for
     * @return      inverse of matrix
     */
    public static float[][] getInverse(float[][] mat)
    {
        float det   = getDet(mat);
        float[][] C = getMatCofactors(mat);
        return scalarMultiply(C, (1/det));
    }

    /**
     * Gets determinant of matrix
     * @param mat   matrix to find determinant of
     * @return      determinant of matrix
     */
    public static float getDet(float[][] mat)
    {
        float ret = 0;
        int dim   = mat.length;

        if (dim > 2)
        {
            float[] toprow = getRow(mat, 0);

            for (int i = 0; i < dim; i++)
            {
                float t         = toprow[i];
                float[][] small = new float[dim - 1][dim - 1];
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
     * @param y     y-value of the row
     * @return      row of matrix
     */
    public static float[] getRow(float[][] mat, int y)
    {
        float[] ret = new float[mat.length];

        for (int i = 0; i < mat.length; i++)
        {
            float[] col = mat[i];
            ret[i]      = col[y];
        }
        return ret;
    }

    /**
     * Gets a column of a matrix
     * @param mat   matrix to get column from
     * @param x     x-value of column
     * @return      column of matrix
     */
    public static float[] getCol(float[][] mat, int x)
    {
        float[] ret = mat[x];
        return ret;
    }

    /**
     * Sets a row of a matrix to given values
     * @param mat   matrix to set row
     * @param row   row to set matrix's row to
     * @param y     y-value of row in matrix
     * @return      matrix with new values in given row
     */
    public static float[][] setRow(float[][] mat, float[] row, int y)
    {
        for (int i = 0; i < mat.length; i++)
        {
            mat[i][y] = row[i];
        }
        return mat;
    }

    /**
     * Sets a column of a matrix to given values
     * @param mat   matrix to set column
     * @param col   column to set matrix's column to
     * @param x     x-value of column in matrix
     * @return      matrix with new values in given column
     */
    public static float[][] setCol(float[][] mat, float[] col, int x)
    {
        mat[x] = col;
        return mat;
    }

    /**
     * Prints a row of a matrix
     * @param mat   matrix to print row of
     * @param y     y-value of row
     */
    public static void printRow(float[][] mat, int y)
    {
        float[] row = getRow(mat, y);
        String out  = "";

        for (float i : row)
        {
            out += i + " ";
        }
        System.out.println(out);
    }

    /**
     * Prints a matrix
     * @param mat   matrix to print
     */
    public static void printMat(float[][] mat)
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
    public static int getX(float[][] mat)
    {
        return mat.length;
    }

    /**
     * Gets y-dimension of matrix
     * @param mat   matrix to get y-dimension of
     * @return      y-dimension of matrix
     */
    public static int getY(float[][] mat)
    {
        return mat[0].length;
    }

    /**
     * Performs sigmoid function on an individual value
     * @param x value
     * @return  transformed "squashed" value
     */
    public static float sigmoidInd(float x)
    {
        return (float) (1 / (1 + Math.pow(Math.E, -x)));
    }

    /**
     * Performs sigmoid prime function on an individual value
     * @param x value
     * @return  transformed value
     */
    public static float sigmoidPrimInd(float x)
    {
        // d/dx((1 + e^-x)^-1) = -1 * (1 + e^-x)^-2 * e^-x * -1 = (1 + e^-x)^-2 * e^-x
        return (float)(Math.pow(1 + Math.pow(Math.E, -x), -2) * (Math.pow(Math.E ,-x)));
    }

    /**
     * Performs sigmoid function on all values in a matrix
     * @param mat   matrix
     * @return      matrix of "squashed" values
     */
    public static float[][] sigmoidMat(float[][] mat)
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

    public static float[][] sigmoidPrimeMat(float[][] mat)
    {
        for (int x = 0; x < getX(mat); x++)
        {
            for (int y = 0; y < getY(mat); y++)
            {
                mat[x][y] = sigmoidPrimInd(mat[x][y]);
            }
        }
        return mat;
    }

    public static float[][] scalarSubMat(float[][] mat1, float[][] mat2)
    {
        if (getX(mat1) != getX(mat2) || getY(mat1) != getY(mat2)) return null;
        float[][] result = new float[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++)
        {
            for (int j = 0; j < mat1[0].length; j++)
            {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        return result;
    }

    public static float[][] scalarAddMat(float[][] mat1, float[][] mat2)
    {
        if (getX(mat1) != getX(mat2) || getY(mat1) != getY(mat2)) return null;
        float[][] result = new float[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++)
        {
            for (int j = 0; j < mat1[0].length; j++)
            {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }

    public static float[][] scalarMultMat(float[][] mat1, float[][] mat2)
    {
        if (getX(mat1) != getX(mat2) || getY(mat1) != getY(mat2)) return null;
        float[][] result = new float[mat1.length][mat1[0].length];
        for (int i = 0; i < mat1.length; i++)
        {
            for (int j = 0; j < mat1[0].length; j++)
            {
                result[i][j] = mat1[i][j] * mat2[i][j];
            }
        }
        return result;
    }

    public static float sumOfAll(float[] mat)
    {
        float sum = 0;
        for (float i : mat)
        {
            sum += i;
        }
        return sum;
    }

    public static float[] scalarMultiply(float[] mat, float scalar)
    {
        for (int i = 0; i < mat.length; i++)
        {
            mat[i] = scalar * mat[i];
        }
        return mat;
    }
}
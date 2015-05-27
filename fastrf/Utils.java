package ca.ubc.cs.beta.models.fastrf;

import java.util.*;

public class Utils {    
    public static int sum(int[] arr) {
        if (arr == null) return 0;
        int l = arr.length;
        int res = 0;
        for (int i=0; i < l; i++) {
            res += arr[i];
        }
        return res;
    }
    
    public static double sum(double[] arr) {
        if (arr == null) return 0;
        int l = arr.length;
        double res = 0;
        for (int i=0; i < l; i++) {
            res += arr[i];
        }
        return res;
    }
    
    public static double mean(double[] arr) {
        if (arr == null) return 0;
        return sum(arr)/arr.length;
    }
    
    public static double var(double[] arr) {
        if (arr == null) return 0;
        int l = arr.length;
        if (l <= 1) return 0;
        
        double sum = 0, sumSq = 0;
        for (int i=0; i < l; i++) {
            sum += arr[i];
            sumSq += arr[i] * arr[i];
        }
        return (sumSq - sum*sum/l)/(l-1);
    }
    
    // TODO: implement O(N) median if we can.
    public static double median(double[] arr) {
        if (arr == null) return Double.NaN;
        int l = arr.length;
        if (l == 0) return Double.NaN;
        Arrays.sort(arr);
        return arr[(int)Math.floor(l/2.0)] / 2 + arr[(int)Math.ceil(l/2.0)] / 2;
    }
    
    public static double prod(double[] arr, int start, int end) {
        double result = 1;
        for (int i=start; i < end; i++) {
            result *= arr[i];
        }
        return result;
    }
}
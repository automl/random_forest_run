package ca.ubc.cs.beta.models.rf;

import java.util.*;
import Weibull.*;
import com.mathworks.toolbox.javabuilder.MWException;
import com.mathworks.toolbox.javabuilder.MWNumericArray;

public class WeibullFit {
    private static WeibullDist weibullDist = null;
    static double[] fit_dist_and_sample(double[] y_pred, boolean[] cens_pred, double[] weights, int numSamples, double lowerBoundForSamples, double maxValueForAllCens) {
        try {
            WeibullDist weibullDist = getWeibullDist();
            y_pred = Arrays.copyOf(y_pred, y_pred.length);
            for (int i=0; i < y_pred.length; i++) if (y_pred[i] == 0) y_pred[i] = 0.001;

            double ALPHA = 0.05; // 95% confidence interval
            Object[] result = weibullDist.wblfit(1, y_pred, ALPHA, cens_pred, weights);
            MWNumericArray output = ((MWNumericArray)result[0]);
            double[] parmhat = output.getDoubleData();
            output.dispose();

            double[] samples;

            for (int i=0; i < parmhat.length; i++) {
                if (Double.isNaN(parmhat[i])) {
                    int numcensored = 0;
                    for (int j=0; j < cens_pred.length; j++) {
                        if (cens_pred[j]) numcensored++;
                    }
                    int numuncensored = cens_pred.length - numcensored;
                    System.out.println("WARNING: cannot fit Weibull to " + numuncensored + " uncensored points and " + numcensored + " censored points.");

                    samples = new double[numSamples];
                    if (numuncensored > 0) {
                        double[] y_uncensored = new double[numuncensored];
                        for (int j=0, k=0; j < cens_pred.length; j++) {
                            if (!cens_pred[j]) y_uncensored[k++]=y_pred[j];
                        }
                        for (int j=0; j < numSamples; j++) {
                            samples[j] = y_uncensored[(int)Math.floor(Math.random() * numuncensored)];
                        }
                    } else {
                        for (int j=0; j < numSamples; j++) {
                            samples[j] = maxValueForAllCens;
                        }
                    }
                    return samples;
                }
            }
            // Weibull successfully fit.
            result = weibullDist.wblcdf(1, lowerBoundForSamples, parmhat[0], parmhat[1]);
            output = ((MWNumericArray)result[0]);
            double lb_quantile = output.getDouble();
            output.dispose();
            double[] u = new double[numSamples];
            for (int i=0; i < numSamples; i++) {
                u[i] = lb_quantile + (1-lb_quantile) * Math.random();
            }
            result = weibullDist.wblinv(1, u, parmhat[0], parmhat[1]);
            output = ((MWNumericArray)result[0]);
            samples = output.getDoubleData();
            output.dispose();
            
            boolean badSamples = false;
            for (int i=0; i < numSamples; i++) {
                if (Double.isNaN(samples[i])) {
                    badSamples = true;
                    break;
                }
            }
            if (badSamples) {
                // if problems with stratified sampling, just sample directly
                System.out.println("WARNING: inverse sampling did not work - sampling directly.");
                for (int j=0; j < numSamples; j++) {
                    if (lb_quantile > 1-1e-5) {
                        samples[j] = maxValueForAllCens;
                    }
                    while (true) {
                        result = weibullDist.wblrnd(1, parmhat[0], parmhat[1], 1, 1);
                        output = ((MWNumericArray)result[0]);
                        samples[j] = output.getDouble();
                        output.dispose();
                        output = ((MWNumericArray)weibullDist.wblcdf(1, samples[j], parmhat[0], parmhat[1])[0]);
                        double new_lb_quantile = output.getDouble();
                        output.dispose();
                        if (new_lb_quantile >= lb_quantile) {
                            break;
                        }
                    }
                }
            }
            
            for (int i=0; i < numSamples; i++) {
                if (samples[i] < lowerBoundForSamples) throw new RuntimeException("Hallucinated samples below lower bound!");
            }
            
            return samples;
        }
        catch (MWException e) {
            throw new RuntimeException(e);
        }
    }
    private static WeibullDist getWeibullDist() throws MWException {
        if (weibullDist == null) {
            weibullDist = new WeibullDist();
        }
        return weibullDist;
    }
}
using System;

namespace NeuralNetwork {

    public static class Functions {

        /*
            Resource:
            https://deepai.org/machine-learning-glossary-and-terms/relu
        */

        public static class ReLU{
            public static double Calculate(double x){
                return Math.Max(0,x);
            }

            public static double Derivative(double x){
                if(x<=0){
                    return 0;
                } else {return 1;}
            }
        }


        /*
            https://paperswithcode.com/method/leaky-relu
        */
        public static class LeakyReLU{

             public static double Calculate(double x){
                if(x<=0){
                    return x*0.01;
                } else {
                    return x;
                }
            }

            public static double Derivative(double x){
                if(x<=0){
                    return 0.01;
                }
                else {
                    return 1;
                }
            }

        }

        /*
            https://en.wikipedia.org/wiki/Hyperbolic_functions
        */
        public static class Tanh {
            
             public static double  Calculate(double x){
                return Math.Tanh(x);
            }

             public static  double Derivative(double x){
                return 1-Math.Tanh(x)*Math.Tanh(x);
            }
        }


        /*
            https://en.wikipedia.org/wiki/Sigmoid_function
        */

        public static class Sigmoid{
             public static double Calculate(double x){
                return 1/(1+Math.Exp(-x));
            }
             public static double Derivative(double x){
                return 1/(2+Math.Exp(x)+Math.Exp(-x));
            }
        }
    }
}
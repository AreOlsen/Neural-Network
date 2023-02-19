using System;

namespace NeuralNetwork{
        public class InputPoint{
        public double[] inputs;
        public double[] expectOutputs;
        public InputPoint(double[] inputs, double[] expectOutputs){
            this.inputs=inputs;
            this.expectOutputs=expectOutputs;
        }
    }
}
(* ::Package:: *)

(* 
  TODO:
  - Initialisation policies
  - Work out the policy for the sizes that ensure all possible DNF expressions
    can be learned. Ten change parameter to [0, 1] from 0 capacity to full capacity to represent
    all possible DNF expressions. Ignoring NOTs then andSize does not need to be larger than 2^(inputSize - 1)
  - XOR neuron (and possibility of extending idea to create a more efficient Majority neuron)
*)

(* ------------------------------------------------------------------ *)
BeginPackage["neurallogic`"]
(* ------------------------------------------------------------------ *)

Harden::usage = "Hards soft bits.";
Soften::usage = "Soften hard bits.";
HardNeuralNOT::usage = "Neural NOT.";
HardNeuralAND::usage = "Hard neural AND.";
HardNeuralNAND::usage = "Hard neural NAND.";
HardNeuralOR::usage = "Hard neural OR.";
HardNeuralNOR::usage = "Hard neural NOR.";
HardNeuralPortLayer::usage = "Port layer.";
AppendHardClassificationLoss::usage = "Append hard classification loss.";
NeuralOR::usage = "Neural OR.";
NeuralAND::usage = "Neural AND.";
SoftAND::usage = "Hard AND.";
SoftOR::usage = "Soft OR.";
SoftNOT::usage = "Soft NOT.";
HardClip::usage = "Hard clip.";
LogisticClip::usage = "Logistic clip.";
HardNeuralMajority::usage = "Hard neural majority.";
HardNeuralChain::usage = "Hard neural chain.";
HardNOT::usage = "Hard NOT.";
HardNAND::usage = "Hard NAND.";
HardNOR::usage = "Hard NOR.";
HardAND::usage = "Hard AND.";
HardOR::usage = "Hard OR.";
HardMajority::usage = "Hard majority.";
ExtractWeights::usage = "Extract weights.";

(* ------------------------------------------------------------------ *)
Begin["`Private`"]
(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
(* Boolean utilities *)
(* ------------------------------------------------------------------ *)

Harden[softBit_] := If[softBit > 0.5, True, False]
Harden[softBits_List] := Map[Harden, softBits]

Soften[hardBit_] := If[hardBit == 1, 1.0, 0.0]
Soften[hardBits_List] := Map[Soften, hardBits]

HardClip[x_] := Clip[x, {0.00000001, 0.9999999}]
LogisticClip[x_] := LogisticSigmoid[4(2 x - 1)]

(* ------------------------------------------------------------------ *)
(* Initalization policies *)
(* ------------------------------------------------------------------ *)

InitializeBalanced[net_] := NetInitialize[net,
  Method -> {"Random", "Weights" -> UniformDistribution[{0.4, 0.6}]}
]

InitializeBiasToZero[net_] := NetInitialize[net,
  Method -> {"Random", "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]]}
]

InitializeBiasToOne[net_] := NetInitialize[net,
  Method -> {"Random", "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[1, 1]]}
]

(* ------------------------------------------------------------------ *)
(* Hard NOT *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => NOT is fully active
  w = 1 => NOT is fully inactive
  Hence, corresponding hard logic is: (b && w) || (! b && ! w)
    or equivalently ! (b \[Xor] w)
*)
SoftNOT[input_, weights_] := 1 - weights + input (2 weights - 1)

HardNOT[{input_List, weights_List}] := 
  {
    Not /@ Thread[Xor[input, First[weights]]],
    Drop[weights, 1]
  }

HardNeuralNOT[inputSize_] := {
  NetGraph[
    <|
      "Weights" -> NetArrayLayer["Output" -> inputSize],
      "WeightsClip" -> ElementwiseLayer[HardClip], 
      "Not" -> ThreadingLayer[SoftNOT[#Input, #Weights] &, 1],
      "OutputClip" -> ElementwiseLayer[LogisticClip] 
    |>,
    {
      "Weights" -> "WeightsClip",
      "WeightsClip" -> NetPort["Not", "Weights"],
      "Not" -> "OutputClip"
    }
  ],
  HardNOT
}

(* ------------------------------------------------------------------ *)
(* Hard AND *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => AND is fully inactive
  w = 1 => AND is fully active
  Hence, corresponding hard logic is: b || !w
*)
SoftAND[b_, w_] := 
  If[w > 1/2,
    If[b > 1/2,
      b,
      (2 w - 1) b + 1 - w
    ], (*else w<=1/2*)
    If[b > 1/2,
      - 2 w (1 - b) + 1,
      1 - w
    ]
  ]

HardAND[{input_List, weights_List}] :=
  {
    Block[{notWeights = (Not /@ #) & /@ First[weights]},
      MapApply[And, Map[Thread[Or[input, #]] &, notWeights]]
    ],
    Drop[weights, 1]
  }

HardNeuralAND[inputSize_, layerSize_] := {
  NetGraph[
    <|
      "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
      "WeightsClip" -> ElementwiseLayer[HardClip],
      "HardInclude" -> ThreadingLayer[SoftAND[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
      "Min" -> AggregationLayer[Min],
      "OutputClip" -> ElementwiseLayer[LogisticClip] 
    |>,
    {
      "Weights" -> "WeightsClip",
      "WeightsClip" -> NetPort["HardInclude", "Weights"],
      "HardInclude" -> "Min",
      "Min" -> "OutputClip"
    }
  ],
  HardAND
}

(* ------------------------------------------------------------------ *)
(* Hard NAND *)
(* ------------------------------------------------------------------ *)

HardNAND[{input_List, weights_List}] := HardNOT[HardAND[{input, weights}]]

HardNeuralNAND[inputSize_, layerSize_] := With[
  {
    neuralAND = HardNeuralAND[inputSize, layerSize],
    neuralNOT = HardNeuralNOT[layerSize]
  },
  With[
    {
      softNeuralAND = First[neuralAND], softNeuralNOT = First[neuralNOT]
    },
    {
      InitializeBiasToZero[NetGraph[
        <|
          "AND" -> softNeuralAND,
          "NOT" -> softNeuralNOT
        |>,
        {
          "AND" -> "NOT"
        }
      ]],
      HardNAND
    }
  ]
]

(* ------------------------------------------------------------------ *)
(* Hard OR *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => OR is fully inactive
  w = 1 => OR is fully active
  Hence, corresponding hard logic is: b && w
*)
SoftOR[b_, w_] := 1 - SoftAND[1-b, w]

HardOR[{input_List, weights_List}] :=
  {
    MapApply[Or, Map[Thread[And[input, #]] &, First[weights]]],
    Drop[weights, 1]
  }

HardNeuralOR[inputSize_, layerSize_] := {
  NetGraph[
    <|
      "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
      "WeightsClip" -> ElementwiseLayer[HardClip],
      "HardInclude" -> ThreadingLayer[SoftOR[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
      "Max" -> AggregationLayer[Max],
      "OutputClip" -> ElementwiseLayer[LogisticClip]
    |>,
    {
      "Weights" -> "WeightsClip",
      "WeightsClip" -> NetPort["HardInclude", "Weights"],
      "HardInclude" -> "Max",
      "Max" -> "OutputClip"
    }
  ],
  HardOR
}

(* ------------------------------------------------------------------ *)
(* Hard NOR *)
(* ------------------------------------------------------------------ *)

HardNOR[{input_List, weights_List}] := HardNOT[HardOR[{input, weights}]]

HardNeuralNOR[inputSize_, layerSize_] := With[
  {
    neuralOR = HardNeuralOR[inputSize, layerSize],
    neuralNOT = HardNeuralNOT[layerSize]
  },
  With[
    {
      softNeuralOR = First[neuralOR], softNeuralNOT = First[neuralNOT]
    },
    {
      InitializeBiasToZero[NetGraph[
        <|
          "OR" -> softNeuralOR,
          "NOT" -> softNeuralNOT
        |>,
        {
          "OR" -> "NOT"
        }
      ]],
      HardNOR
    }
  ]
]

(* ------------------------------------------------------------------ *)
(* Hard MAJORITY *)
(* ------------------------------------------------------------------ *)

HardMajority[{input_List, weights_List}] := 
  {
    Block[{weightedInputs = Map[HardNOT[{input, #}] &, First[weights]]},
      MapApply[Majority, weightedInputs]
    ],
    Drop[weights, 1]
  }

(* 
  Currently using sort (probably compiles to QuickSort)
    - average O(n log n)
    - worst-case O(n^2)
  Replace with Floyd-Rivest algorithm:
    - average n + \min(k, n - k) + O(\sqrt{n \log n}) comparisons with probability at least 1 - 2n^{-1/2}
    - worst-case O(n^2)
    - See https://danlark.org/2020/11/11/miniselect-practical-and-generic-selection-algorithms/
*)
HardNeuralMajority[inputSize_, layerSize_] := {
  With[{medianIndex = Ceiling[(inputSize + 1)/2]},
    InitializeBalanced[NetGraph[
      <|
        "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
        "WeightsClip" -> ElementwiseLayer[HardClip],
        "HardInclude" -> ThreadingLayer[SoftNOT[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
        "Sort" -> FunctionLayer[Sort /@ # &],
        "Medians" -> PartLayer[{All, medianIndex}],
        "OutputClip" -> ElementwiseLayer[LogisticClip]
      |>,
      {
        "Weights" -> "WeightsClip",
        "WeightsClip" -> NetPort["HardInclude", "Weights"],
        "HardInclude" -> "Sort",
        "Sort" -> "Medians",
        "Medians" -> "OutputClip"
      }
    ]]
  ],
  HardMajority
}

(* ------------------------------------------------------------------ *)
(* Hard port layer *)
(* ------------------------------------------------------------------ *)

HardNeuralPortLayer[inputSize_, numPorts_] := {
  ReshapeLayer[{numPorts, inputSize / numPorts}],
  Function[{input},
    {
      Partition[First[input], numPorts], 
      Last[input]
    }
  ]
}

(* ------------------------------------------------------------------ *)
(* Hard neural chain *)
(* ------------------------------------------------------------------ *)

HardNeuralChain[layers_List] := Module[{chain = Transpose[layers]},
  With[
    {
      softChain = First[chain],
      hardChain = Last[chain]
    },
    {
      NetChain[softChain],
      RightComposition @@ hardChain
    }
  ]
]

(* ------------------------------------------------------------------ *)
(* Hard classification loss *)
(* ------------------------------------------------------------------ *)

AppendHardClassificationLoss[net_] := NetGraph[
  <|
    "NeuralLogicNet" -> net,
    "Probs" -> AggregationLayer[Mean],
    "loss" -> MeanSquaredLossLayer[]
  |>,  
  {
    "NeuralLogicNet" -> "Probs",
    "Probs" -> NetPort["loss", "Input"],
    "loss" -> NetPort["Loss"]
  }
]

(* ------------------------------------------------------------------ *)
(* Approximate differentiable AND, OR *)
(* ------------------------------------------------------------------ *)

NeuralAND[inputSize_, layerSize_] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
    "WeightsClip" -> ElementwiseLayer[HardClip],
    "SoftInclude" -> ThreadingLayer[1 - #Weights (1 - #Input) &, 1, "Output" -> {layerSize, inputSize}],
    (* LogSumExp trick *)
    "And1" -> ElementwiseLayer[Log],
    "And2" -> AggregationLayer[Total],
    "And3" -> ElementwiseLayer[Exp],
    "OutputClip" -> ElementwiseLayer[HardClip] 
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "And1",
    "And1" -> "And2",
    "And2" -> "And3",
    "And3" -> "OutputClip"
  }
]

NeuralOR[inputSize_, layerSize_] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
    "WeightsClip" -> ElementwiseLayer[HardClip],
    "SoftInclude" -> ThreadingLayer[1 - #Weights #Input &, 1, "Output" -> {layerSize, inputSize}],
    (* LogSumExp trick *) 
    "Or1" -> ElementwiseLayer[Log],
    "Or2" -> AggregationLayer[Total],
    "Or3" -> ElementwiseLayer[Exp],
    "Or4" -> ElementwiseLayer[1 - # &],
    "OutputClip" -> ElementwiseLayer[HardClip]
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "Or1",
    "Or1" -> "Or2",
    "Or2" -> "Or3",
    "Or3" -> "Or4",
    "Or4" -> "OutputClip"
  }
]

(* ------------------------------------------------------------------ *)
(* Network hardening *)
(* ------------------------------------------------------------------ *)

ExtractWeights[net_] := Module[{weights, arrays},
  weights = Select[Quiet[NetExtract[net, {All, "Weights"}]], ! MissingQ[#] &];
  arrays = NetExtract[#, "Arrays"]["Array"] & /@ weights;
  Normal /@ arrays
]

End[]

EndPackage[]

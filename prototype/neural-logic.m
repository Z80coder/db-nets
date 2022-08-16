(* ::Package:: *)

(* 
  TODO:
  - 0.5 causes a numerical error hardening problem. Can we fix?
  - Generalise COUNT to BooleanCountingFunction and specialise to XOR etc. Also consider
    explicit XOR neuron (and possibility of extending idea to create a more efficient Majority neuron)
  - Initialisation policies
  - Work out the policy for the sizes that ensure all possible DNF expressions
    can be learned. Then change parameter to [0, 1] from 0 capacity to full capacity to represent
    all possible DNF expressions. Ignoring NOTs then andSize does not need to be larger than 2^(inputSize - 1)
*)

(* ------------------------------------------------------------------ *)
BeginPackage["neurallogic`"]
(* ------------------------------------------------------------------ *)

Harden::usage = "Hards soft bits.";
Soften::usage = "Soften hard bits.";
HardNeuralAND::usage = "Hard neural AND.";
HardNeuralNAND::usage = "Hard neural NAND.";
HardNeuralOR::usage = "Hard neural OR.";
HardNeuralNOR::usage = "Hard neural NOR.";
HardNeuralReshapeLayer::usage = "Port layer.";
NeuralOR::usage = "Neural OR.";
NeuralAND::usage = "Neural AND.";
DifferentiableHardAND::usage = "Differentiable hard AND.";
DifferentiableHardOR::usage = "Differentiable hard OR.";
HardClip::usage = "Hard clip.";
LogisticClip::usage = "Logistic clip.";
HardNeuralMajority::usage = "Hard neural majority.";
HardNeuralChain::usage = "Hard neural chain.";
HardNAND::usage = "Hard NAND.";
HardNOR::usage = "Hard NOR.";
HardAND::usage = "Hard AND.";
HardOR::usage = "Hard OR.";
HardMajority::usage = "Hard majority.";
GetWeights::usage = "Get weights.";
GetNetArrays::usage = "Get net arrays.";
ExtractWeights::usage = "Extract weights.";
HardNetFunction::usage = "Hard net function.";
HardNetTransformWeights::usage = "Hard net transform weights.";
HardNetBooleanExpression::usage = "Hard net boolean expression.";
HardNetBooleanFunction::usage = "Hard net boolean function.";
HardenNet::usage = "Hard net.";
HardClassificationLoss::usage = "Hard classification loss.";
InitializeNearToZero::usage = "Initialize bias to zero.";
InitializeNearToOne::usage = "Initialize bias to one.";
InitializeBalanced::usage = "Initialize balanced.";
InitializeToConstant::usage = "Initialize to constant.";
HardeningLayer::usage = "Hardening layer.";
HardeningForward::usage = "Hardening forward.";
HardeningBackward::usage = "Hardening backward.";
HardNeuralCount::usage = "Hard neural count.";
HardNeuralExactlyK::usage = "Hard neural exactly k.";
HardNeuralLTEK::usage = "Hard neural less than or equal to k.";
Require::usage = "Require.";
HardDropoutLayer::usage = "Hard dropout layer.";
RandomUniformSoftBits::usage = "Random soft bit layer.";
RandomNormalSoftBits::usage = "Random soft bit layer.";
RandomBalancedNormalSoftBits::usage = "Random balanced normal soft bit layer.";
SoftBits::usage = "Create some soft bits.";
HardNetClassBits::usage = "Hard net class bits.";
HardNetClassScores::usage = "Hard net class scores.";
HardNetClassProbabilities::usage = "Hard net class probabilities.";
HardNetClassPrediction::usage = "Hard net class prediction.";
HardNetClassify::usage = "Hard net classify.";
HardNetClassifyEvaluation::usage = "Hard net classify evaluation.";
DifferentiableHardNOT::usage = "Differentiable hard NOT.";
HardNeuralNOT::usage = "Neural NOT.";

(* ------------------------------------------------------------------ *)

Begin["`Private`"]

(* ------------------------------------------------------------------ *)
(* Boolean utilities *)
(* ------------------------------------------------------------------ *)

Harden[softBit_] := If[softBit > 0.5, True, False]
Harden[softBits_List] := Harden /@ softBits

Soften[hardBit_] := If[hardBit == 1, 1.0, 0.0]
Soften[hardBits_List] := Map[Soften, hardBits]

HardClip[x_, d_:0.0] := Clip[x, {d, 1.0 - d}]
LogisticClip[x_] := LogisticSigmoid[4(2 x - 1)]

(* ------------------------------------------------------------------ *)
(* Initalization policies *)
(* ------------------------------------------------------------------ *)

InitializeToConstant[net_, k_] := NetInitialize[net, All,
  Method -> {
    "Random", 
    "Weights" -> UniformDistribution[{k, k}],
    "Biases" -> UniformDistribution[{k, k}]
  }
]

InitializeBalanced[net_] := NetInitialize[net, All,
  Method -> {
    "Random", 
    "Weights" -> UniformDistribution[{0.4, 0.6}],
    "Biases" -> UniformDistribution[{0.4, 0.6}]
  }
]

InitializeNearToZero[net_] := NetInitialize[net, All,
  Method -> {
    "Random", 
    "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]],
    "Biases" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]]
  }
]

InitializeNearToOne[net_] := NetInitialize[net, All,
  Method -> 
  {
    "Random", 
    "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[1, 1]],
    "Biases" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[1, 1]]
  }
]

(* ------------------------------------------------------------------ *)
(* Hardening layer *)
(* ------------------------------------------------------------------ *)

HardeningForward[] := Function[
  {Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 2]]},
  If[# > 0.5, 1.0, 0.0] & /@ # & /@ input
]

HardeningBackward[] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 2]],
    Typed[outgrad, TypeSpecifier["PackedArray"]["MachineReal", 2]]
  },
  (* Straight-through estimator *)
  outgrad
]

HardeningLayer[] := CompiledLayer[HardeningForward[], HardeningBackward[]]

(* ------------------------------------------------------------------ *)
(* Learnable soft-bit deterministic variables *)
(* ------------------------------------------------------------------ *)

SoftBits[array_NetArrayLayer] := NetGraph[
  <|
    "SoftBits" -> NetGraph[
      <|
        "Weights" -> array,
        "Clip" -> ElementwiseLayer[HardClip]
      |>,
      {
        "Weights" -> "Clip"
      }
    ]
  |>,
  {}
]

SoftBits[size_] := SoftBits[NetArrayLayer["Output" -> size]]

BalancedSoftBits[size_] := InitializeBalanced[SoftBits[size]]
NearZeroSoftBits[size_] := InitializeNearToZero[SoftBits[size]]

(* ------------------------------------------------------------------ *)
(* Learnable soft-bit random variables *)
(* ------------------------------------------------------------------ *)

RandomUniformSoftBits[aWeights_List, bWeights_List] := NetGraph[
  <|
    "RandomUniformSoftBits" -> NetGraph[
      <|
        "A" -> NetArrayLayer["Array" -> aWeights, "Output" -> Length[aWeights]],
        "B" -> NetArrayLayer["Array" -> aWeights, "Output" -> Length[aWeights]],
        "ClipA" -> ElementwiseLayer[HardClip],
        "ClipB" -> ElementwiseLayer[HardClip],
        "Distribution" -> RandomArrayLayer[UniformDistribution[{0, 1}], "Output" -> Length[aWeights]],
        "Variates" -> FunctionLayer[#A + (#B - #A) #Random &]
      |>,
      {
        "Distribution" -> NetPort["Variates", "Random"],
        "A" -> "ClipA",
        "B" -> "ClipB",
        "ClipA" -> NetPort["Variates", "A"],
        "ClipB" -> NetPort["Variates", "B"]
      }
    ]
  |>,
  {}
]

RandomUniformSoftBits[size_] := RandomUniformSoftBits[Table[RandomReal[{0.0, 1.0}], size], Table[RandomReal[{0.0, 1.0}, size]]]

RandomNormalSoftBits[muWeights_List, sigmaWeights_List] := NetGraph[
  <|
    "RandomNormalSoftBits" -> NetGraph[
      <|
        "Mu" -> NetArrayLayer["Array" -> muWeights, "Output" -> Length[muWeights]],
        "Sigma" -> NetArrayLayer["Array" -> sigmaWeights, "Output" -> Length[muWeights]],
        "ClipMu" -> ElementwiseLayer[HardClip],
        "ClipSigma" -> ElementwiseLayer[Clip[#, {0.0, Infinity}] &],
        "Distribution" -> RandomArrayLayer[NormalDistribution[0, 1], "Output" -> Length[muWeights]],
        "Variates" -> FunctionLayer[#Mu + #Sigma * #Random &],
        "ClipVariates" -> ElementwiseLayer[HardClip]
      |>,
      {
        "Distribution" -> NetPort["Variates", "Random"],
        "Mu" -> "ClipMu",
        "ClipMu" -> NetPort["Variates", "Mu"],
        "Sigma" -> "ClipSigma",
        "ClipSigma" -> NetPort["Variates", "Sigma"],
        "Variates" -> "ClipVariates"
      }
    ]
  |>,
  {}
]

RandomNormalSoftBits[size_] := RandomNormalSoftBits[Table[RandomReal[{0, 0.5}], size], Table[RandomReal[{0, 0.1}], size]]
RandomBalancedNormalSoftBits[size_] := RandomNormalSoftBits[Table[RandomReal[{0, 1}], size], Table[RandomReal[{0, 0.1}], size]]

(* ------------------------------------------------------------------ *)
(* Hard NOT *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => NOT is fully active
  w = 1 => NOT is fully inactive
  Hence, corresponding hard logic is: (b && w) || (! b && ! w)
    or equivalently ! (b \[Xor] w)
*)
DifferentiableHardNOT[input_, weights_] := 1 - weights + input (2 weights - 1)

HardNOT[input_, weight_] := Not[Xor[input, weight]]

HardNOT[input_/;VectorQ[input], weights_/;VectorQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardNOT[input_/;VectorQ[input], weights_/;VectorQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardNOT[input_/;VectorQ[input], weights_/;VectorQ[weights]] weight dimensions"];*)
  ConfirmAssert[Length[input] == Length[weights], Null, "HardNOT[input_/;VectorQ[input], weights_/;VectorQ[weights]] semantics check"];
  Map[HardNOT[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]

HardNOT[input_/;MatrixQ[input], weights_/;VectorQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardNOT[input_/;MatrixQ[input], weights_/;VectorQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardNOT[input_/;MatrixQ[input], weights_/;VectorQ[weights]] weight dimensions"];*)
  HardNOT[#, weights] & /@ input
]

HardNOT[input_/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardNOT[input_/;VectorQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardNOT[input_/;VectorQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  HardNOT[input, #] & /@ weights
]

HardNOT[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardNOT[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardNOT[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  HardNOT[input, weights]
]

HardNOT[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardNOT[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardNOT[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  ConfirmAssert[Length[input] == Length[weights], Null, "HardNOT[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] semantics check"];
  Map[HardNOT[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]

HardNOT[layerSize_] := Function[{inputs},
  Block[{input, weights, layerWeights, reshapedWeights, output},
    {input, weights} = inputs;
    (*Echo[Dimensions[input],"HardNOT[layerSize_] input dimensions"];*)
    {
      (* Output *)
      layerWeights = First[weights];
      reshapedWeights = Partition[layerWeights, Length[layerWeights] / layerSize];
      output = HardNOT[input, reshapedWeights];
      (*Echo[Dimensions[input],"HardNOT[layerSize_] output dimensions"];*)
      output,
      (* Consume weights *)
      Drop[weights, 1]
    }
  ]
]

HardNeuralNOT[inputSize_, layerSize_, weights_Function:BalancedSoftBits] := {
  NetGraph[
    <|
      "Weights" -> weights[layerSize * inputSize],
      "Reshape" -> ReshapeLayer[{layerSize, inputSize}],
      "Not" -> ThreadingLayer[DifferentiableHardNOT[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}]
    |>,
    {
      "Weights" -> "Reshape",
      "Reshape" -> NetPort["Not", "Weights"]
    } 
  ],
  HardNOT[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard AND *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => AND is fully inactive
  w = 1 => AND is fully active
  Hence, corresponding hard logic is: b || !w
*)
DifferentiableHardAND[b_, w_] := Max[b, 1 - w]

HardAND[input_, weight_] := Or[input, Not[weight]]

HardAND[input_/;VectorQ[input], weights_/;VectorQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardAND[input_/;VectorQ[input], weights_/;VectorQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardAND[input_/;VectorQ[input], weights_/;VectorQ[weights]] weight dimensions"];*)
  ConfirmAssert[Length[input] == Length[weights], Null, "HardAND[input_/;VectorQ[input], weights_/;VectorQ[weights]] semantics check"];
  (* This is a reduce operation *)
  And @@ Map[HardAND[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]

HardAND[input_/;MatrixQ[input], weights_/;VectorQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardAND[input_/;MatrixQ[input], weights_/;VectorQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardAND[input_/;MatrixQ[input], weights_/;VectorQ[weights]] weight dimensions"];*)
  HardAND[#, weights] & /@ input
]

HardAND[input_/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardAND[input_/;VectorQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardAND[input_/;VectorQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  HardAND[input, #] & /@ weights  
]

HardAND[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardAND[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardAND[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  HardAND[input, weights]
]

HardAND[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardAND[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardAND[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  ConfirmAssert[Length[input] == Length[weights], Null, "HardAND[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] semantics check"];
  Map[HardAND[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]

HardAND[layerSize_] := Function[{inputs},
  Block[{input, weights, layerWeights, reshapedWeights, output},
    {input, weights} = inputs;
    (*Echo[Dimensions[input],"HardAND[layerSize_] input dimensions"];*)
    {
      (* Output *)
      layerWeights = First[weights];
      reshapedWeights = Partition[layerWeights, Length[layerWeights] / layerSize];
      output = HardAND[input, reshapedWeights];
      (*Echo[Dimensions[input],"HardAND[layerSize_] output dimensions"];*)
      output,
      (* Consume weights *)
      Drop[weights, 1]
    }
  ]
]

HardNeuralAND[inputSize_, layerSize_, weights_Function:NearZeroSoftBits] := {
  NetGraph[
    <|
      "Weights" -> weights[layerSize * inputSize],
      "Reshape" -> ReshapeLayer[{layerSize, inputSize}],
      "HardInclude" -> ThreadingLayer[DifferentiableHardAND[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
      "And" -> AggregationLayer[Min]
    |>,
    {
      "Weights" -> "Reshape",
      "Reshape" -> NetPort["HardInclude", "Weights"],
      "HardInclude" -> "And"
    }
  ],
  HardAND[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard NAND *)
(* ------------------------------------------------------------------ *)

HardNAND[layerSize_] := Function[{inputs},
  HardNOT[1][HardAND[layerSize][inputs]]
]

HardNeuralNAND[inputSize_, layerSize_, andWeights_Function:NearZeroSoftBits, notWeights_Function:BalancedSoftBits] := {
  NetChain[
    {
      First[HardNeuralAND[inputSize, layerSize, andWeights[#]&]],
      First[HardNeuralNOT[layerSize, 1, notWeights[#]&]]
    }
  ],
  HardNAND[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard OR *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => OR is fully inactive
  w = 1 => OR is fully active
  Hence, corresponding hard logic is: b && w
*)
DifferentiableHardOR[b_, w_] := 1 - DifferentiableHardAND[1-b, w]

HardOR[input_, weight_] := And[input, weight]

HardOR[input_/;VectorQ[input], weights_/;VectorQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardOR[input_/;VectorQ[input], weights_/;VectorQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardOR[input_/;VectorQ[input], weights_/;VectorQ[weights]] weight dimensions"];*)
  ConfirmAssert[Length[input] == Length[weights], Null, "HardOR[input_/;VectorQ[input], weights_/;VectorQ[weights]] semantics check"];
  (* This is a reduce operation *)
  Or @@ Map[HardOR[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]

HardOR[input_/;MatrixQ[input], weights_/;VectorQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardOR[input_/;MatrixQ[input], weights_/;VectorQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardOR[input_/;MatrixQ[input], weights_/;VectorQ[weights]] weight dimensions"];*)
  HardOR[#, weights] & /@ input
]

HardOR[input_/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardOR[input_/;VectorQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardOR[input_/;VectorQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  HardOR[input, #] & /@ weights  
]

HardOR[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardOR[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardOR[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  HardOR[input, weights]
]

HardOR[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] := Block[{},
  (*Echo[Dimensions[input],"HardOR[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] input dimensions"];*)
  (*Echo[Dimensions[weights],"HardOR[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] weight dimensions"];*)
  ConfirmAssert[Length[input] == Length[weights], Null, "HardOR[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] semantics check"];
  Map[HardOR[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]

HardOR[layerSize_] := Function[{inputs},
  Block[{input, weights, layerWeights, reshapedWeights, output},
    {input, weights} = inputs;
    (*Echo[Dimensions[input],"HardOR[layerSize_] input dimensions"];*)
    {
      (* Output *)
      layerWeights = First[weights];
      reshapedWeights = Partition[layerWeights, Length[layerWeights] / layerSize];
      output = HardOR[input, reshapedWeights];
      (*Echo[Dimensions[input],"HardOR[layerSize_] output dimensions"];*)
      output,
      (* Consume weights *)
      Drop[weights, 1]
    }
  ]
]

HardNeuralOR[inputSize_, layerSize_, weights_Function:BalancedSoftBits] := {
  NetGraph[
    <|
      "Weights" -> weights[layerSize * inputSize],
      "Reshape" -> ReshapeLayer[{layerSize, inputSize}],
      "HardInclude" -> ThreadingLayer[DifferentiableHardOR[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
      "Or" -> AggregationLayer[Max]
    |>,
    {
      "Weights" -> "Reshape",
      "Reshape" -> NetPort["HardInclude", "Weights"],
      "HardInclude" -> "Or"
    }
  ],
  HardOR[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard NOR *)
(* ------------------------------------------------------------------ *)

HardNOR[layerSize_] := Function[{inputs},
  HardNOT[1][HardOR[layerSize][inputs]]
]

HardNeuralNOR[inputSize_, layerSize_, orWeights_Function:NearZeroSoftBits, notWeights_Function:BalancedSoftBits] := {
  NetChain[
    {
      First[HardNeuralOR[inputSize, layerSize, orWeights[#]&]],
      First[HardNeuralNOT[layerSize, 1, notWeights[#]&]]
    }
  ],
  HardNOR[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard MAJORITY *)
(* ------------------------------------------------------------------ *)

HardMajority[] := Function[{inputsAndWeights},
  Block[{inputs, weights, output},
    {inputs, weights} = inputsAndWeights;
    {
      (* Output *)
      Map[
        Block[{input = #},
          ConfirmAssert[ListQ[input], Null, "HardMajority semantics check"];
          Majority @@ input
        ] &,
        inputs
      ],
      (* Don't consume weights *)
      weights
    }
  ]
]

(* TODO: remove need to specify inputSize *)
HardNeuralMajority[numInputs_, inputSize_] := {
  With[{medianIndex = Floor[(inputSize + 1)/2]},
    NetGraph[
      <|
        "Sort" -> FunctionLayer[Sort /@ # &],
        "Medians" -> PartLayer[{All, medianIndex}, "Output" -> numInputs]
      |>,
      {
        "Sort" -> "Medians"
      }
    ]
  ],
  HardMajority[]
}

(* ------------------------------------------------------------------ *)
(* Hard COUNT *)
(* Experimental *)
(* ------------------------------------------------------------------ *)

(* TODO: Simplify with Ordering layer *)
HardNeuralCount[numArrays_, arraySize_] := {
  NetGraph[
    <|
      "Sort" -> FunctionLayer[
        NumericalSort /@ # &
      ],
      "DropLast" -> FunctionLayer[
        Part[#, 1 ;; arraySize - 1] & /@ # &
      ],
      "PadFalse" -> FunctionLayer[
        ArrayPad[#, {{1, 0}}] & /@ # &,
        "Output" -> {numArrays, arraySize}
      ],
      "CountBooleans" -> FunctionLayer[
        (* !a && b *)
        MapThread[Min[DifferentiableHardNOT[#1, 0], #2] &, {#Input2, #Input1}, 2] &
      ](*,
      "OutputClip" -> ElementwiseLayer[LogisticClip]*)
    |>,
    {
      "Sort" -> NetPort["CountBooleans", "Input1"],
      "Sort" -> "DropLast",
      "DropLast" -> "PadFalse",
      "PadFalse" -> NetPort["CountBooleans", "Input2"](*,
      "CountBooleans" -> "OutputClip"*)
    }
  ],
  (* TODO: implement this *)
  HardCount
}

HardNeuralExactlyK[numArrays_, arraySize_, k_] := {
  NetGraph[
    <|
      "Count" -> HardNeuralCount[numArrays, arraySize][[1]],
      "SelectK" -> FunctionLayer[
        Part[#, arraySize - k + 1] & /@ # &
      ]
    |>,
    {
      "Count" -> "SelectK"
    }
  ],
  (* TODO: implement this *)
  HardExactlyK
}

HardNeuralLTEK[numArrays_, arraySize_, k_] := {
  NetGraph[
    <|
      "Count" -> HardNeuralCount[numArrays, arraySize][[1]],
      "CountsLTEK" -> FunctionLayer[
        Part[#, arraySize - k + 1 ;; arraySize] & /@ # &
      ],
      "LTEK" -> AggregationLayer[Max]
    |>,
    {
      "Count" -> "CountsLTEK",
      "CountsLTEK" -> "LTEK"
    }
  ],
  (* TODO: implement this *)
  LTEK
}

Require[requirement_] := {
  NetGraph[
    <|
      "Requirement" -> requirement,
      "Require" -> ThreadingLayer[
        Min[#K, #Input] &,
        2
      ]
    |>,
    {
      "Requirement" -> NetPort["Require", "K"]
    }
  ],
  (* TODO: implement this *)
  Require
}

(* ------------------------------------------------------------------ *)
(* Hard Dropout layer *)
(* ------------------------------------------------------------------ *)

HardDropout[{input_List, weights_List}] := {input, weights}

HardDropoutLayer[p_] := {
  NetGraph[
    <|
      "Dropout" -> DropoutLayer[p, "OutputPorts" -> "BinaryMask"],
      "Mask" -> FunctionLayer[#BinaryMask #Input &]
    |>,
    {
      NetPort["Input"] -> NetPort["Dropout", "Input"],
      NetPort["Input"] -> NetPort["Mask", "Input"],
      NetPort["Dropout", "BinaryMask"] -> NetPort["Mask", "BinaryMask"]
    }
  ],
  HardDropout
}

(* ------------------------------------------------------------------ *)
(* Hard reshape layer *)
(* ------------------------------------------------------------------ *)

HardReshapeLayer[numPorts_] := Function[{inputs},
  Block[{input, weights},
    {input, weights} = inputs;
    input = Flatten[input];
    {
      Partition[input, Length[input] / numPorts], 
      weights
    }
  ]
]

HardNeuralReshapeLayer[inputSize_, numPorts_] := {
  ReshapeLayer[{numPorts, inputSize / numPorts}],
  HardReshapeLayer[numPorts]
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

(*
  TODO: CrossEntropy seems better but need to do a full
  experimental comparison.
*)
HardClassificationLoss[] := NetGraph[
  <|
    "Harden" -> HardeningLayer[],
    "Mean" -> AggregationLayer[Mean, 2],
    "Error" -> MeanSquaredLossLayer[]
  |>,
  {
    "Harden" -> "Mean",
    "Mean" -> NetPort["Error", "Input"]
  } 
]

HardClassificationLoss[] := NetGraph[
  <|
    "Harden" -> HardeningLayer[],
    "SoftProbs" -> AggregationLayer[Total, 2],
    "SoftmaxLayer" -> SoftmaxLayer[],
    "Error" -> CrossEntropyLossLayer["Probabilities"]
  |>,
  {
    "Harden" -> "SoftProbs",
    "SoftProbs" -> "SoftmaxLayer",
    "SoftmaxLayer" -> NetPort["Error", "Input"]
  } 
]

(* ------------------------------------------------------------------ *)
(* Network hardening *)
(* ------------------------------------------------------------------ *)

MakeArrayPath[netArrayName_] := Map[
  If[NumberQ[ToExpression[#]], ToExpression[#], #] &, 
  StringSplit[Normal[netArrayName]["Name"], "/"]
]

UnknownSoftBitType[arrays_List] := Nothing

HardenSoftBits[arrays_List] := Map[
  Block[{arrayName = First[#], array = Normal[Last[#]]},
    MakeArrayPath[arrayName] -> NumericArray[Boole[Harden[array]], "UnsignedInteger8"]
  ] &, 
  arrays
]

HardenRandomSoftBits[randomVariate_, p1_, p2_] := Block[
  {
    p1ArrayName = MakeArrayPath[First[p1]],
    p1Array = Normal[Last[p1]],
    p2ArrayName = MakeArrayPath[First[p2]],
    p2Array = Normal[Last[p2]],
    parameters,
    realisedSoftBits,
    realisedHardBits,
    replacementWeights
  },
  parameters = Partition[Riffle[p1Array, p2Array], 2];
  realisedSoftBits = Map[randomVariate[#] &, parameters];
  realisedHardBits = NumericArray[Boole[Harden[realisedSoftBits]], "UnsignedInteger8"];
  replacementWeights = SoftBits[NetArrayLayer["Array" -> realisedHardBits]];
  Take[p1ArrayName, First[FirstPosition[p1ArrayName, "Weights"]]] -> replacementWeights
]

HardenRandomSoftBits[hardener_, arrays_List] := Block[{paramPairs},
  (* Assume arrays are correctly ordered *)
  paramPairs = Partition[arrays, 2];
  Map[Block[{p1, p2},
      {p1, p2} = #;
      hardener[p1, p2]
    ] &,
    paramPairs
  ]
]

HardenRandomNormalSoftBits[mus_, sigmas_] := HardenRandomSoftBits[
  With[{mu = HardClip[#[[1]]], sigma = Clip[#[[2]], {0.0, Infinity}]},
    mu + sigma RandomVariate[NormalDistribution[0, 1]]
  ] &,
  mus,
  sigmas
]
HardenRandomNormalSoftBits[arrays_List] := HardenRandomSoftBits[HardenRandomNormalSoftBits, arrays]

HardenRandomUniformSoftBits[as_, bs_] := HardenRandomSoftBits[
  With[{a = HardClip[#[[1]]], b = HardClip[#[[2]]]},
    a + (b - a) RandomVariate[UniformDistribution[{0, 1}]]
  ] &, 
  as, 
  bs
]
HardenRandomUniformSoftBits[arrays_List] := HardenRandomSoftBits[HardenRandomUniformSoftBits, arrays]

HardenNet[net_] := Module[
  {
    weights = NetExtract[NetInsertSharedArrays[net], NetArray[All]],
    hardenings,
    replacements
  },
  hardenings = GroupBy[
    Normal[weights],
    With[{key = MakeArrayPath[First[#]]},
      If[MemberQ[key, "SoftBits"], HardenSoftBits,
        If[MemberQ[key, "RandomUniformSoftBits"], HardenRandomUniformSoftBits,
          If[MemberQ[key, "RandomNormalSoftBits"], HardenRandomNormalSoftBits, UnknownSoftBitType]
        ]
      ]
    ] &
  ];
  replacements = Flatten[KeyValueMap[#1[#2] &, hardenings]];
  NetReplacePart[net, replacements]
]

GetNetArrays[net_] := Select[Normal[NetFlatten[net]], MatchQ[#, _NetArrayLayer] &]

GetWeights[net_] := NetExtract[#, "Arrays"]["Array"] & /@ GetNetArrays[net]
 
ExtractWeights[net_] := Normal[Values[GetWeights[net]]]

HardNetFunction[hardNet_, trainedSoftNet_] := Module[{softWeights},
  softWeights = ExtractWeights[trainedSoftNet];
  With[{hardWeights = Harden[softWeights]},
    Function[{input},
      First[hardNet[{input, hardWeights}]]
    ]
  ]
]

HardNetBooleanExpression[hardNetFunction_Function, inputSize_] := Module[
  {
    inputs = Table[Symbol["b" <> ToString[i]], {i, inputSize}]
  },
  hardNetFunction[inputs]
]

HardNetBooleanFunction[hardNetBooleanExpression_, inputSize_] := Block[
  {
    signature = Typed[Symbol["input"], TypeSpecifier["NumericArray"]["MachineInteger", 1]],
    replacements = Quiet[Table[
      Symbol["b" <> ToString[i]] -> With[{b = Symbol["input"][[i]]}, 
          If[b == 1, True, False]
        ],
      {i, inputSize}]
    ],
    indexExpression
  },
  indexExpression = hardNetBooleanExpression //. replacements;
  Function[Evaluate[signature], Evaluate[indexExpression]]
]

(* ------------------------------------------------------------------ *)
(* Classifier querying and evaluation *)
(* ------------------------------------------------------------------ *)

HardNetClassBits[hardNet_Function, extractInput_, data_] := Normal[hardNet[Harden[Normal[extractInput[#]]]] & /@ data]

HardNetClassScores[classBits_] := (Total /@ Boole[#]) & /@ classBits

HardNetClassProbabilities[classScores_] := N[Exp[#]/Total[Exp[#]]] & /@ classScores

HardNetClassPrediction[classProbabilities_, decoder_] := First[decoder @ classProbabilities]

HardNetClassify[hardNet_Function, data_, decoder_:(# &), extractInput_:(#["Input"] &), extractTarget_:(#["Target"] &)] := 
  Association /@ ResourceFunction["DynamicMap"][
    {
      "Prediction" -> HardNetClassPrediction[
          HardNetClassProbabilities[
            HardNetClassScores[
              HardNetClassBits[hardNet, extractInput, {#}]
            ]
          ],
          decoder
        ],
      "Target" -> extractTarget[#]
    } &,
    data(*,
    Method -> "CoarsestGrained"*)
  ]

HardNetClassifyEvaluation[hardNetClassify_] := Module[
  {results = Counts[hardNetClassify], correctResults, totalCorrect, totalResults, accuracy},
  correctResults = KeySelect[results, Length[DeleteDuplicates[Values[#]]] == 1 &];
  totalCorrect = Total[correctResults];
  totalResults = Total[results];
  accuracy = N[totalCorrect / totalResults];
  <|"Accuracy" -> accuracy, "Results" -> Reverse[NumericalSort[results]]|>
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
    "Log" -> ElementwiseLayer[Log],
    "Sum" -> AggregationLayer[Total],
    "Exp" -> ElementwiseLayer[Exp],
    "OutputClip" -> ElementwiseLayer[HardClip] 
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "Log",
    "Log" -> "Sum",
    "Sum" -> "Exp",
    "Exp" -> "OutputClip"
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

End[]

EndPackage[]

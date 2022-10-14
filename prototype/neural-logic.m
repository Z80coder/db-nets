(* ::Package:: *)

(* 
  Notes:
  - Indication that removing flat regimes in gradient improves performance. Revisit all the piecewise functions.
  - Even with perfect soft-hard translation semantics that hard boundary at 0.5, plus numerical error in floating point
  representation, means that hardening can introduce deviations between soft and hard performance. The simplest fix is
  to train the soft-net with 64, rather than 32, bit precision.
  - BooleanCountingFunction and XOR neurons not yet fully implemented.
  - Determine the all-purpose logic layer that will work for most problems (e.g. DNF, CNF, etc.)
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
HardNeuralXOR::usage = "Hard neural XOR.";
HardNeuralReshapeLayer::usage = "Port layer.";
HardNeuralCatenateLayer::usage = "Catenate layer.";
HardNeuralFlattenLayer::usage = "Flatten layer.";
NeuralOR::usage = "Neural OR.";
NeuralAND::usage = "Neural AND.";
DifferentiableHardAND::usage = "Differentiable hard AND.";
DifferentiableHardOR::usage = "Differentiable hard OR.";
DifferentiableHardXOR::usage = "Differentiable hard XOR.";
HardClip::usage = "Hard clip.";
LogisticClip::usage = "Logistic clip.";
MajorityIndex::usage = "Majority index.";
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
NeuralHardeningLayer::usage = "Neural hardening layer.";
HardeningForward::usage = "Hardening forward.";
HardeningBackward::usage = "Hardening backward.";
HardDropoutLayer::usage = "Hard dropout layer.";
RandomUniformSoftBits::usage = "Random soft bit layer.";
RandomNormalSoftBits::usage = "Random soft bit layer.";
RandomBalancedNormalSoftBits::usage = "Random balanced normal soft bit layer.";
SoftBits::usage = "Create some soft bits.";
BalancedSoftBits::usage = "Create some balanced soft bits.";
NearZeroSoftBits::usage = "Create some near zero soft bits.";
HardNetClassBits::usage = "Hard net class bits.";
HardNetClassScores::usage = "Hard net class scores.";
HardNetClassProbabilities::usage = "Hard net class probabilities.";
HardNetClassPrediction::usage = "Hard net class prediction.";
HardNetClassify::usage = "Hard net classify.";
HardNetClassifyEvaluation::usage = "Hard net classify evaluation.";
DifferentiableHardNOT::usage = "Differentiable hard NOT.";
HardNeuralNOT::usage = "Neural NOT.";
HardNeuralCount::usage = "Hard neural count.";
HardNeuralExactlyK::usage = "Hard neural exactly k.";
HardNeuralLTEK::usage = "Hard neural less than or equal to k.";
Require::usage = "Require.";
RealEncoderDecoder::usage = "Real encoder decoder.";
RealTo1Hot::usage = "Real to nonlinear one hot.";
BinaryCountToReal::usage = "Binary count to real.";
HardNeuralRealLayer::usage = "Binary count to real layer.";
HardNeuralANDorOR::usage = "Hard neural AND or OR.";
DifferentiableHardIfThenElse::usage = "Differentiable hard if then else.";
DifferentiableHardIfThenElse1::usage = "Differentiable hard if then else version 2.";
DifferentiableHardIfThenElse2::usage = "Differentiable hard if then else version 3.";
HardIfThenElse::usage = "Hard if then else.";
HardNeuralIfThenElseLayer::usage = "If then else layer.";
BlendFactor::usage = "Blend factor.";
HardNeuralDecisionList::usage = "Hard neural decision list.";
ConditionAction::usage = "Condition action.";
ConditionActionLayers::usage = "Condition action layers.";
OpenActions::usage = "Open actions.";

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
  },
  RandomSeeding -> Automatic
]

InitializeBalanced[net_] := NetInitialize[net, All,
  Method -> {
    "Random", 
    "Weights" -> UniformDistribution[{0.4, 0.6}],
    "Biases" -> UniformDistribution[{0.4, 0.6}]
  },
  RandomSeeding -> Automatic
]

InitializeNearToZero[net_] := NetInitialize[net, All,
  Method -> {
    "Random", 
    "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]],
    "Biases" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]]
  },
  RandomSeeding -> Automatic
]

InitializeNearToOne[net_] := NetInitialize[net, All,
  Method -> 
  {
    "Random", 
    "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[1, 1]],
    "Biases" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[1, 1]]
  },
  RandomSeeding -> Automatic
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

(* Specify size to operate faster at batch level *)
HardeningLayer[size_] := CompiledLayer[HardeningForward[], HardeningBackward[], "Input" -> size]

NeuralHardeningLayer[] := {HardeningLayer[], Identity[#] &}

(* Specify size to operate faster at batch level *)
NeuralHardeningLayer[size_] := {HardeningLayer[size], Identity[#] &}

(* ------------------------------------------------------------------ *)
(* Learnable soft-bit deterministic variables *)
(* ------------------------------------------------------------------ *)

(* TODO: soft and hard weights should be named, and the hard weights represented as
an association (rather than consuming nested lists). This would avoid problems
due to softNet evaluation order differing from hardNet evaluation order, and would
also support weight re-use in the hard function. *)
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
(* TODO: rename to Mask *)
DifferentiableHardAND[b_, w_] := Max[b, 1 - w]

(*
  This version slower.
  TODO: compare to above
  DifferentiableHardAND[b_, w_] := If[w > 1/2, If[b > 1/2, b, (2w -1)b + 1 - w], If[b > 1/2, -2w(1 - b) + 1, 1 - w]]
*)

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
(* TODO: rename to Mask *)
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
  Block[{inputs, weights},
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

MajorityIndex[inputSize_] := Ceiling[(inputSize + 1) / 2]

(* inputSize is the length of each array passed to Majority *)
HardNeuralMajority[inputSize_] := {
  With[{majorityIndex = MajorityIndex[inputSize]},
    NetGraph[
      <|
        "Sort" -> FunctionLayer[ReverseSort /@ # &],
        "Medians" -> PartLayer[{All, majorityIndex}]
      |>,
      {
        "Sort" -> "Medians"
      }
    ]
  ],
  HardMajority[]
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

(* TODO: remove *)
HardReshapeLayerDeprecated[numPorts_] := Function[{inputs},
  Block[{input, weights},
    {input, weights} = inputs;
    input = Flatten[input];
    {
      Partition[input, Length[input] / numPorts], 
      weights
    }
  ]
]

(* TODO: remove *)
HardNeuralReshapeLayerDeprecated[inputSize_, numPorts_] := {
  ReshapeLayer[{numPorts, inputSize / numPorts}],
  HardReshapeLayer[numPorts]
}

HardReshapeLayer[dims_] := Function[{inputs},
  Block[{input, weights},
    {input, weights} = inputs;
    {
      ArrayReshape[input, dims],
      weights
    }
  ]
]

HardNeuralReshapeLayer[dims_] := {
  ReshapeLayer[dims],
  HardReshapeLayer[dims]
}

(* ------------------------------------------------------------------ *)
(* Hard catenate layer *)
(* ------------------------------------------------------------------ *)

HardCatenateLayer[] := Function[{inputs},
  Block[{input, weights},
    {input, weights} = inputs;
    input = Flatten[input];
    {
      input, 
      weights
    }
  ]
]

HardNeuralCatenateLayer[] := {
  CatenateLayer[],
  HardCatenateLayer[]
}

(* ------------------------------------------------------------------ *)
(* Hard flatten layer *)
(* ------------------------------------------------------------------ *)

HardNeuralFlattenLayer[] := {
  FlattenLayer[],
  (* N.B. Same as Catenate in hard case *)
  HardCatenateLayer[]
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
(* Classification utilities *)
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
(* Regression utilities *)
(* ------------------------------------------------------------------ *)

(* TODO:
A problem with binary count decoding is that extreme values are under-represented (in terms of
number of bit vectors that map to them). E.g. there's only 1 way to have max bits, but n
ways to have 1 bit. This is a problem for regression, where we want to be able to represent
the full range of values. One solution is to use a different encoding, e.g. Gray code.
*)

(* Binary count decoding *)
BinaryCountToReal[{min_, max_}] := Function[{input},
  Block[{y},
    y = (Total[input] * 1.0) / Length[input];
    y = (max - min) y + min;
    {Clip[y, {min, max}]}
  ]
]

HardBinaryCountToReal[{min_, max_}] := Function[{inputs},
  Block[{input, weights, output},
    {input, weights} = inputs;
    input = Flatten[input];
    output = (Total[Boole[input]] * 1.0) / Length[input];
    output = (max - min) output + min;
    output = {Clip[output, {min, max}]};
    {
      output,
      (* Don't consume weights *)
      weights
    }
  ]
]

RealTo1Hot[realValues_List, sampleRate_] := Module[{uniqueValues, min, max},
  uniqueValues = DeleteDuplicates[realValues];
  {min, max} = MinMax[uniqueValues];
  uniqueValues = RandomSample[uniqueValues, Round[Length[uniqueValues] * sampleRate]];
  uniqueValues = DeleteDuplicates[Join[{min}, uniqueValues, {max}]];
  uniqueValues = Sort[uniqueValues];
  With[{v = uniqueValues, len = Length[uniqueValues]},
    { 
      Function[{x},
        Block[{dists, minPosition},
          dists = Map[Abs[x - #] &, v];
          minPosition = First[Ordering[dists, 1]];
          Table[If[i == minPosition, 1, 0], {i, 1, len}]
        ]
      ],
      Function[{b},
        Block[{position},
          position = First[FirstPosition[b, 1]];
          If[MissingQ[position],
            Last[v],
            v[[position]]
          ]
        ]
      ],
      v
    }
  ]
] 

RealEncoderDecoder[realValues_, sampleRate_] := Module[{min, max, encoder, decoder, numBits},
  {min, max} = MinMax[realValues];
  {encoder, decoder, uniqueValues} = RealTo1Hot[realValues, sampleRate];
  numBits = Length[uniqueValues];
  Association[{ 
    "NumBits" -> numBits,
    "MinMax" -> {min, max},
    "DecoderFunction" -> decoder,
    "NetEncoder" -> NetEncoder[{"Function", encoder, {numBits}}]
  }]
]

HardNeuralRealLayer[{min_, max_}] := {
  NetGraph[
    <|
      "HardeningLayer" -> HardeningLayer[],
      "BinaryCountToReal" -> FunctionLayer[BinaryCountToReal[{min, max}]]
    |>,
    {
      "HardeningLayer" -> "BinaryCountToReal"
    }
  ],
  HardBinaryCountToReal[{min, max}]
}

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

GetNetArrays[normalNet_List] := Select[normalNet, MatchQ[#, _NetArrayLayer] &]

GetNetArrays[normalNet_Association] := GetNetArrays[Values[normalNet]]

GetNetArrays[net_] := GetNetArrays[Normal[NetFlatten[net]]]

GetWeights[net_] := NetExtract[#, "Arrays"]["Array"] & /@ GetNetArrays[net]
 
ExtractWeights[net_] := Normal[GetWeights[net]] 

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
    signature = Typed[Symbol["input"], TypeSpecifier["ListVector"]["MachineInteger", 1]],
    replacements = Quiet[Table[
      Symbol["b" <> ToString[i]] -> With[{b = Symbol["b"][[i]]}, 
          (*If[b == 1, True, False]*)
          b
        ],
      {i, inputSize}]
    ],
    indexExpression
  },
  indexExpression = hardNetBooleanExpression //. replacements;
  With[{expr = indexExpression},
    Function[
      Evaluate[signature],
      Block[{b},
        b = If[# == 1, True, False] & /@ input;
        expr
      ]
    ]
  ]
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
    data  
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

(* ------------------------------------------------------------------ *)
(* Experimental *)
(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
(* If-Then-Else *)
(* ------------------------------------------------------------------ *)

HardIfThenElse[condition_, ifTrue_, ifFalse_] := Block[{},
  (*
  Echo["HardIfThenElse[condition, ifTrue, ifFalse]"];
  Echo[condition, "condition"];
  Echo[ifTrue, "ifTrue"];
  Echo[ifFalse, "ifFalse"];
  *)
  If[condition, ifTrue, ifFalse]
]

HardIfThenElse[c_/;VectorQ[c], b1_/;VectorQ[b1], b2_/;VectorQ[b2]] := Block[{actionsPerCondition, pb1, pb2},
  (*
  Echo["HardIfThenElse[c_/;VectorQ[c], b1_/;VectorQ[b1], b2_/;VectorQ[b2]]"];
  Echo[c, "c"];
  Echo[b1, "b1"];
  Echo[b2, "b2"];
  *)
  ConfirmAssert[Length[b1] == Length[b2], Null, "The action sizes must be identical"];
  ConfirmAssert[Length[b1] >= Length[c], Null, "The size of the action must be greater than equal to the size of the condition"];
  ConfirmAssert[IntegerQ[Length[b1]/Length[c]], Null, "The size of action must be a multiple of the size of the condition"];
  actionsPerCondition = Length[b1]/Length[c];
  pb1 = Partition[b1, actionsPerCondition];
  pb2 = Partition[b2, actionsPerCondition];
  Flatten[MapThread[HardIfThenElse, {c, pb1, pb2}], 1]
]

HardIfThenElse[c_/;VectorQ[c], b1_/;MatrixQ[b1], b2_/;MatrixQ[b2]] := Block[{},
  (*
  Echo["HardIfThenElse[c_/;VectorQ[c], b1_/;MatrixQ[b1], b2_/;MatrixQ[b2]]"];
  Echo[c, "c"];
  Echo[b1, "b1"];
  Echo[b2, "b2"];
  *)
  ConfirmAssert[Dimensions[b1] == Dimensions[b2], Null, "The action dimensions must be identical"];
  ConfirmAssert[Length[b1] >= Length[c], Null, "The size of the action must be greater than equal to the size of the condition"];
  ConfirmAssert[IntegerQ[Length[b1]/Length[c]], Null, "The size of action must be a multiple of the size of the condition"];
  MapThread[HardIfThenElse, {c, b1, b2}]
]

(*
HardOR[input_/;MatrixQ[input], weights_/;VectorQ[weights]] := Block[{},
  HardOR[#, weights] & /@ input
]

HardOR[input_/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  HardOR[input, #] & /@ weights  
]

HardOR[{input_}/;VectorQ[input], weights_/;MatrixQ[weights]] := Block[{},
  HardOR[input, weights]
]

HardOR[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] := Block[{},
  ConfirmAssert[Length[input] == Length[weights], Null, "HardOR[input_/;MatrixQ[input], weights_/;MatrixQ[weights]] semantics check"];
  Map[HardOR[#[[1]], #[[2]]] &, Partition[Riffle[input, weights], 2]]
]
*)

(* TODO: we can't guarantee the order of the extracted weights, and thefore the implicit order in this
function may mismatch the extracted weights. We need to switch from consuming weights to accessing
named weights.    *)
ComputeHardIfThenElse[condition_, ifTrue_, ifFalse_] := Function[{inputs},
  Block[{input, weights, conditionValue, ifTrueValue, ifFalseValue},
    (*
    Echo[condition, "ComputeHardIfThenElse"];
    Echo[condition, "condition"];
    Echo[ifTrue, "ifTrue"];
    Echo[ifFalse, "ifFalse"]; 
    *)
    {input, weights} = (*Echo[*)inputs(*, "inputs"]*);
    (*Echo["weight dimensions: " <> ToString[Dimensions[weights]] <> ": " <> ToString[Dimensions/@weights]];*)
    (*Echo["Evaluating IfFalseFunction"];*)
    {ifFalseValue, weights} = (*Echo[*)ifFalse[{input, weights}](*, "return value from IfFalseFunction"]*);
    (*Echo["Evaluating IfTrueFunction"];*)
    {ifTrueValue, weights} = (*Echo[*)ifTrue[{input, weights}](*, "return value from ifTrueFunction"]*);
    (*Echo["Evaluating conditionFunction"];*)
    {conditionValue, weights} = (*Echo[*)condition[{input, weights}](*, "return value from conditionFunction"]*);
    {
      (* Output *)
      HardIfThenElse[conditionValue, ifTrueValue, ifFalseValue],
      (* Weights *)
      weights
    }
  ]
]

(* Hack to avoid ordering problem for now *)
ComputeHardIfThenElse2[condition_, ifTrue_, ifFalse_] := Function[{inputs},
  Block[{input, weights, conditionValue, ifTrueValue, ifFalseValue},
    (*
    Echo[condition, "ComputeHardIfThenElse"];
    Echo[condition, "condition"];
    Echo[ifTrue, "ifTrue"];
    Echo[ifFalse, "ifFalse"]; 
    *)
    {input, weights} = (*Echo[*)inputs(*, "inputs"]*);
    (*Echo["weight dimensions: " <> ToString[Dimensions[weights]] <> ": " <> ToString[Dimensions/@weights]];*)
    (*Echo["Evaluating IfFalseFunction"];*)
    {ifFalseValue, weights} = (*Echo[*)ifFalse[{input, weights}](*, "return value from IfFalseFunction"]*);
    (*Echo["Evaluating conditionFunction"];*)
    {conditionValue, weights} = (*Echo[*)condition[{input, weights}](*, "return value from conditionFunction"]*);
    (*Echo["Evaluating IfTrueFunction"];*)
    {ifTrueValue, weights} = (*Echo[*)ifTrue[{input, weights}](*, "return value from ifTrueFunction"]*);
    {
      (* Output *)
      HardIfThenElse[conditionValue, ifTrueValue, ifFalseValue],
      (* Weights *)
      weights
    }
  ]
]

(* Simple *)
DifferentiableHardIfThenElse1[w_, b1_, b2_] := Max[Min[b1, w], Min[b2, 1 - w]]

(* More complex *)
DifferentiableHardIfThenElse2[w_, b1_, b2_] := If[
  b1 <= 1/2 && b2 <= 1/2,
    If[w > 1/2,
      If[b2 > b1, 2 (b1 - b2) w + 2 b2 - b1, b1],
      If[b2 > b1, b2, 2 (b1 - b2) w + b2]
    ],
    If[b1 > 1/2 && b2 > 1/2,
      If[w > 1/2,
        If[b2 <= b1, 2 (b1 - b2) w + 2 b2 - b1, b1],
        If[b2 <= b1, b2, 2 (b1 - b2) w + b2]
      ],
      If[w > 1/2,
        (2 b1 - 1) w + 1 - b1,
        (1 - 2 b2) w + b2
      ]
    ]
  ]

DifferentiableHardIfThenElse[w_, b1_, b2_] := DifferentiableHardIfThenElse1[w, b1, b2]

HardNeuralIfThenElseLayer[] := {
  FunctionLayer[DifferentiableHardIfThenElse[#Condition, #IfTrue, #IfFalse] &],
  ComputeHardIfThenElse[#1, #2, #3] &
}

OpenActions[] := {
  FunctionLayer[DifferentiableHardIfThenElse[#Condition, #IfTrue, #IfFalse] &],
  ComputeHardIfThenElse2[#1, #2, #3] &
}

(* TOOD: this is a hack because softWeights don't follow the {softNet, hardNet} format*)
GetWeightStub[] := Function[{inputs},
  Block[{input, weights},
    {input, weights} = inputs;
    {
      Take[weights, 1],
      Drop[weights, 1]
    }
  ]
]

ConditionAction[condition_, action_] := Module[{fullCondition, fullAction, conditionOutputSize, actionOutputSize},
  (* Some condition/actions will be {softNet, hardNet} format; others will just be softNet *)
  (* TODO: fix this *)
  fullCondition = If[ListQ[condition], condition, {condition, GetWeightStub[]}];
  fullAction = If[ListQ[action], action, {action, GetWeightStub[]}];
  conditionOutputSize = NetExtract[First[fullCondition], "Output"];
  actionOutputSize = NetExtract[First[fullAction], "Output"];
  ConfirmAssert[actionOutputSize >= conditionOutputSize, Null, "The size of the action must be greater than equal to the size of the condition"];
  ConfirmAssert[IntegerQ[actionOutputSize/conditionOutputSize], Null, "The size of action must be a multiple of the size of the condition"];
  {
    fullCondition,
    HardNeuralChain[{fullAction, HardNeuralReshapeLayer[{conditionOutputSize, actionOutputSize / conditionOutputSize}]}]
  }
]

ConditionActionLayers[conditionActions_List, defaultAction_, iteLayer_:HardNeuralIfThenElseLayer] := Module[
  {layers, lastCondition, reshapedDefaultAction},
  layers = Reverse[
    Map[
      Block[{condition = #[[1]], action = #[[2]], softCondition, hardCondition, softAction, hardAction, softITE, hardITE},
        {softITE, hardITE} = iteLayer[];
        {{softCondition, hardCondition}, {softAction, hardAction}} = ConditionAction[condition, action];
        {
          NetGraph[
            <|
              "Condition" -> softCondition,
              "IfTrue" -> softAction,
              "IfThenElse" -> softITE
            |>,
            {
              "Condition" -> NetPort["IfThenElse", "Condition"],
              "IfTrue" -> NetPort["IfThenElse", "IfTrue"]
            }
          ],
          With[{hardITELiteral = hardITE, hardConditionLiteral = hardCondition, hardActionLiteral = hardAction},
            hardITELiteral[hardConditionLiteral, hardActionLiteral, #1] &
          ]
        }
      ] &, 
      conditionActions
    ]
  ];
  lastCondition = First[Last[conditionActions]];
  reshapedDefaultAction = Last[ConditionAction[lastCondition, defaultAction]];
  {reshapedDefaultAction, layers} 
]

HardNeuralDecisionList[conditionActionLayers_] := Module[
  {layers, softLayers, hardLayers, softReshapedDefaultAction, hardReshapedDefaultAction},
  {{softReshapedDefaultAction, hardReshapedDefaultAction}, layers} = conditionActionLayers;
  softLayers = First /@ layers;
  hardLayers = Last /@ layers;
  {
    Fold[
      NetGraph[
        {#1, #2},
        {NetPort[1, "Output"] -> NetPort[2, "IfFalse"]}
      ] &, 
      softReshapedDefaultAction, 
      softLayers
    ],    
    Fold[
      Function[{inputs},
        Block[{ifThenFunction, elseFunction, ifThenElseFunction},
          (*Echo["ITE function"];*)
          ifThenFunction = #2;  
          elseFunction = #1;
          ifThenElseFunction = ifThenFunction[elseFunction];
          (*Echo[ifThenElseFunction, "ifThenElseFunction"];*)
          ifThenElseFunction[inputs]  
        ]
      ] &,  
      hardReshapedDefaultAction,
      hardLayers
    ]
  }
]

(* ------------------------------------------------------------------ *)
(* Hard AND-or-OR *)
(* ------------------------------------------------------------------ *)

(* TODO: use if-then-else *)
HardNeuralANDorOR[inputSize_, layerSize_, weights_Function : BalancedSoftBits] := {
  NetGraph[
    <|
      "MaskWeights" -> weights[layerSize*inputSize],
      "Reshape" -> ReshapeLayer[{layerSize, inputSize}],
      "HardInclude1" -> ThreadingLayer[DifferentiableHardAND[#Input, #MaskWeights] &, 1, "Output" -> {layerSize, inputSize}],
      "And" -> AggregationLayer[Min],
      "HardInclude2" -> ThreadingLayer[DifferentiableHardOR[#Input, #MaskWeights] &, 1, "Output" -> {layerSize, inputSize}],
      "Or" -> AggregationLayer[Max],
      "CombineWeights" -> weights[layerSize],
      (* Multiplication doesn't preserve hard semantics *)
      "Combine" -> FunctionLayer[#CombineWeights #And + (1 - #CombineWeights) #Or &]
    |>,
    {
      "MaskWeights" -> "Reshape",
      "Reshape" -> NetPort["HardInclude1", "MaskWeights"],
      "HardInclude1" -> "And",
      "Reshape" -> NetPort["HardInclude2", "MaskWeights"],
      "HardInclude2" -> "Or",
      "CombineWeights" -> NetPort["Combine", "CombineWeights"],
      "And" -> NetPort["Combine", "And"],
      "Or" -> NetPort["Combine", "Or"]
    }
  ],
  HardANDOR[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard XOR *)
(* ------------------------------------------------------------------ *)

(*
  w = 0 => OR is fully inactive
  w = 1 => OR is fully active
  Hence, corresponding hard logic is: b && w
*)
IncludeXOR[b_, w_] := 1 - DifferentiableHardAND[1-b, w]

(*
DifferentiableHardXOR[b_List] := Fold[HardXOR[#1, #2] &, 0.0, b]
*)

(*
  XOR is equivalent to: 
    (! a || ! b) && (a || b)
    ! (! a || b) || ! (a || ! b)
*)
DifferentiableHardXOR[b1_, b2_] := Min[Max[1 - b1, 1 - b2], Max[b1, b2]]
DifferentiableHardXOR[b1_, b2_] := 1 - Max[Max[1 - b1, b2], Max[b1, 1 - b2]]

HardNeuralXOR[inputSize_, layerSize_, weights_Function:BalancedSoftBits] := {
  NetGraph[
    <|
      "Weights" -> weights[layerSize * inputSize],
      "Reshape" -> ReshapeLayer[{inputSize, layerSize}],
      "Include" -> ThreadingLayer[IncludeXOR[#Input, #Weights] &, 2, "Output" -> {inputSize, layerSize}],
      "Xor1" -> NetFoldOperator[FunctionLayer[DifferentiableHardXOR[#Input, #State] &], "Output" -> {inputSize, layerSize}],
      "Xor2" -> SequenceLastLayer[]
      (* Method 2 *)
      (*
      "Transpose" -> TransposeLayer[],
      "Xor1" -> NetFoldOperator[FunctionLayer[DifferentiableHardXOR[#Input, #State] &], "Output" -> {inputSize, layerSize}],
      "Xor2" -> SequenceLastLayer[]
      *)
      (* Method 1 *)
      (*"Xor1" -> FunctionLayer[Fold[DifferentiableHardXOR[#1, #2] &, 0.0, Transpose[#]] &]*)
    |>,
    {
      "Weights" -> "Reshape",
      "Reshape" -> NetPort["Include", "Weights"],
      "Include" -> "Xor1",
      "Xor1" -> "Xor2"
    }
  ],
  HardXOR[layerSize]
}

(* ------------------------------------------------------------------ *)
(* Hard COUNT *)
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

End[]

EndPackage[]

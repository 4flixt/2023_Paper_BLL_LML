��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.12unknown8��
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:(*
dtype0
r
02_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name02_dense/bias
k
!02_dense/bias/Read/ReadVariableOpReadVariableOp02_dense/bias*
_output_shapes
:(*
dtype0
z
02_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((* 
shared_name02_dense/kernel
s
#02_dense/kernel/Read/ReadVariableOpReadVariableOp02_dense/kernel*
_output_shapes

:((*
dtype0
r
01_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name01_dense/bias
k
!01_dense/bias/Read/ReadVariableOpReadVariableOp01_dense/bias*
_output_shapes
:(*
dtype0
z
01_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(* 
shared_name01_dense/kernel
s
#01_dense/kernel/Read/ReadVariableOpReadVariableOp01_dense/kernel*
_output_shapes

:(*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
.
0
1
2
3
#4
$5*
.
0
1
2
3
#4
$5*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 

2serving_default* 

0
1*

0
1*
* 
�
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

8trace_0* 

9trace_0* 
_Y
VARIABLE_VALUE01_dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUE01_dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

@trace_0* 
_Y
VARIABLE_VALUE02_dense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUE02_dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Ftrace_0* 

Gtrace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_201_dense/kernel01_dense/bias02_dense/kernel02_dense/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_867282
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#01_dense/kernel/Read/ReadVariableOp!01_dense/bias/Read/ReadVariableOp#02_dense/kernel/Read/ReadVariableOp!02_dense/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_867471
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename01_dense/kernel01_dense/bias02_dense/kernel02_dense/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_867499��
�
�
__inference__traced_save_867471
file_prefix.
*savev2_01_dense_kernel_read_readvariableop,
(savev2_01_dense_bias_read_readvariableop.
*savev2_02_dense_kernel_read_readvariableop,
(savev2_02_dense_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_01_dense_kernel_read_readvariableop(savev2_01_dense_bias_read_readvariableop*savev2_02_dense_kernel_read_readvariableop(savev2_02_dense_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*G
_input_shapes6
4: :(:(:((:(:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:((: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 
�	
�
B__inference_output_layer_call_and_return_conditional_losses_867089

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_867241
input_2
dense_867224:(
dense_867226:(
dense_867229:((
dense_867231:(
output_867234:(
output_867236:
identity

identity_1�� 01_dense/StatefulPartitionedCall� 02_dense/StatefulPartitionedCall�output/StatefulPartitionedCall�
 01_dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_867224dense_867226*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_01_dense_layer_call_and_return_conditional_losses_867056�
 02_dense/StatefulPartitionedCallStatefulPartitionedCall)01_dense/StatefulPartitionedCall:output:0dense_867229dense_867231*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_02_dense_layer_call_and_return_conditional_losses_867073�
output/StatefulPartitionedCallStatefulPartitionedCall)02_dense/StatefulPartitionedCall:output:0output_867234output_867236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_867089x
IdentityIdentity)02_dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(x

Identity_1Identity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^01_dense/StatefulPartitionedCall!^02_dense/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 01_dense/StatefulPartitionedCall 01_dense/StatefulPartitionedCall2D
 02_dense/StatefulPartitionedCall 02_dense/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�

�
D__inference_02_dense_layer_call_and_return_conditional_losses_867410

inputs0
matmul_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:((*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������(W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
(__inference_model_2_layer_call_fn_867301

inputs
unknown:(
	unknown_0:(
	unknown_1:((
	unknown_2:(
	unknown_3:(
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_867097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_model_2_layer_call_fn_867114
input_2
unknown:(
	unknown_0:(
	unknown_1:((
	unknown_2:(
	unknown_3:(
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_867097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_867370

inputs6
$dense_matmul_readvariableop_resource:(3
%dense_biasadd_readvariableop_resource:(8
&dense_matmul_readvariableop_resource_0:((5
'dense_biasadd_readvariableop_resource_0:(7
%output_matmul_readvariableop_resource:(4
&output_biasadd_readvariableop_resource:
identity

identity_1��01_dense/BiasAdd/ReadVariableOp�01_dense/MatMul/ReadVariableOp�02_dense/BiasAdd/ReadVariableOp�02_dense/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
01_dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0{
01_dense/MatMulMatMulinputs&01_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
01_dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
01_dense/BiasAddBiasAdd01_dense/MatMul:product:0'01_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(b
01_dense/TanhTanh01_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
02_dense/MatMul/ReadVariableOpReadVariableOp&dense_matmul_readvariableop_resource_0*
_output_shapes

:((*
dtype0�
02_dense/MatMulMatMul01_dense/Tanh:y:0&02_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
02_dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0�
02_dense/BiasAddBiasAdd02_dense/MatMul:product:0'02_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(b
02_dense/TanhTanh02_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
output/MatMulMatMul02_dense/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
IdentityIdentity02_dense/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������(h

Identity_1Identityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^01_dense/BiasAdd/ReadVariableOp^01_dense/MatMul/ReadVariableOp ^02_dense/BiasAdd/ReadVariableOp^02_dense/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
01_dense/BiasAdd/ReadVariableOp01_dense/BiasAdd/ReadVariableOp2@
01_dense/MatMul/ReadVariableOp01_dense/MatMul/ReadVariableOp2B
02_dense/BiasAdd/ReadVariableOp02_dense/BiasAdd/ReadVariableOp2@
02_dense/MatMul/ReadVariableOp02_dense/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_02_dense_layer_call_and_return_conditional_losses_867073

inputs0
matmul_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:((*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������(W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
(__inference_model_2_layer_call_fn_867221
input_2
unknown:(
	unknown_0:(
	unknown_1:((
	unknown_2:(
	unknown_3:(
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_867185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�

�
D__inference_01_dense_layer_call_and_return_conditional_losses_867390

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������(W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_867185

inputs
dense_867168:(
dense_867170:(
dense_867173:((
dense_867175:(
output_867178:(
output_867180:
identity

identity_1�� 01_dense/StatefulPartitionedCall� 02_dense/StatefulPartitionedCall�output/StatefulPartitionedCall�
 01_dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_867168dense_867170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_01_dense_layer_call_and_return_conditional_losses_867056�
 02_dense/StatefulPartitionedCallStatefulPartitionedCall)01_dense/StatefulPartitionedCall:output:0dense_867173dense_867175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_02_dense_layer_call_and_return_conditional_losses_867073�
output/StatefulPartitionedCallStatefulPartitionedCall)02_dense/StatefulPartitionedCall:output:0output_867178output_867180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_867089x
IdentityIdentity)02_dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(x

Identity_1Identity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^01_dense/StatefulPartitionedCall!^02_dense/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 01_dense/StatefulPartitionedCall 01_dense/StatefulPartitionedCall2D
 02_dense/StatefulPartitionedCall 02_dense/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_02_dense_layer_call_fn_867399

inputs
unknown:((
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_02_dense_layer_call_and_return_conditional_losses_867073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_867499
file_prefix2
 assignvariableop_01_dense_kernel:(.
 assignvariableop_1_01_dense_bias:(4
"assignvariableop_2_02_dense_kernel:((.
 assignvariableop_3_02_dense_bias:(2
 assignvariableop_4_output_kernel:(,
assignvariableop_5_output_bias:

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_01_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_01_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_02_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_02_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
D__inference_01_dense_layer_call_and_return_conditional_losses_867056

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������(W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_01_dense_layer_call_fn_867379

inputs
unknown:(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_01_dense_layer_call_and_return_conditional_losses_867056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_867429

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
'__inference_output_layer_call_fn_867419

inputs
unknown:(
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_867089o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_867345

inputs6
$dense_matmul_readvariableop_resource:(3
%dense_biasadd_readvariableop_resource:(8
&dense_matmul_readvariableop_resource_0:((5
'dense_biasadd_readvariableop_resource_0:(7
%output_matmul_readvariableop_resource:(4
&output_biasadd_readvariableop_resource:
identity

identity_1��01_dense/BiasAdd/ReadVariableOp�01_dense/MatMul/ReadVariableOp�02_dense/BiasAdd/ReadVariableOp�02_dense/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
01_dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0{
01_dense/MatMulMatMulinputs&01_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
01_dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
01_dense/BiasAddBiasAdd01_dense/MatMul:product:0'01_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(b
01_dense/TanhTanh01_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
02_dense/MatMul/ReadVariableOpReadVariableOp&dense_matmul_readvariableop_resource_0*
_output_shapes

:((*
dtype0�
02_dense/MatMulMatMul01_dense/Tanh:y:0&02_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
02_dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0�
02_dense/BiasAddBiasAdd02_dense/MatMul:product:0'02_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(b
02_dense/TanhTanh02_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
output/MatMulMatMul02_dense/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
IdentityIdentity02_dense/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������(h

Identity_1Identityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^01_dense/BiasAdd/ReadVariableOp^01_dense/MatMul/ReadVariableOp ^02_dense/BiasAdd/ReadVariableOp^02_dense/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
01_dense/BiasAdd/ReadVariableOp01_dense/BiasAdd/ReadVariableOp2@
01_dense/MatMul/ReadVariableOp01_dense/MatMul/ReadVariableOp2B
02_dense/BiasAdd/ReadVariableOp02_dense/BiasAdd/ReadVariableOp2@
02_dense/MatMul/ReadVariableOp02_dense/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_867097

inputs
dense_867057:(
dense_867059:(
dense_867074:((
dense_867076:(
output_867090:(
output_867092:
identity

identity_1�� 01_dense/StatefulPartitionedCall� 02_dense/StatefulPartitionedCall�output/StatefulPartitionedCall�
 01_dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_867057dense_867059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_01_dense_layer_call_and_return_conditional_losses_867056�
 02_dense/StatefulPartitionedCallStatefulPartitionedCall)01_dense/StatefulPartitionedCall:output:0dense_867074dense_867076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_02_dense_layer_call_and_return_conditional_losses_867073�
output/StatefulPartitionedCallStatefulPartitionedCall)02_dense/StatefulPartitionedCall:output:0output_867090output_867092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_867089x
IdentityIdentity)02_dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(x

Identity_1Identity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^01_dense/StatefulPartitionedCall!^02_dense/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 01_dense/StatefulPartitionedCall 01_dense/StatefulPartitionedCall2D
 02_dense/StatefulPartitionedCall 02_dense/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_867261
input_2
dense_867244:(
dense_867246:(
dense_867249:((
dense_867251:(
output_867254:(
output_867256:
identity

identity_1�� 01_dense/StatefulPartitionedCall� 02_dense/StatefulPartitionedCall�output/StatefulPartitionedCall�
 01_dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_867244dense_867246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_01_dense_layer_call_and_return_conditional_losses_867056�
 02_dense/StatefulPartitionedCallStatefulPartitionedCall)01_dense/StatefulPartitionedCall:output:0dense_867249dense_867251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_02_dense_layer_call_and_return_conditional_losses_867073�
output/StatefulPartitionedCallStatefulPartitionedCall)02_dense/StatefulPartitionedCall:output:0output_867254output_867256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_867089x
IdentityIdentity)02_dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(x

Identity_1Identity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^01_dense/StatefulPartitionedCall!^02_dense/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 01_dense/StatefulPartitionedCall 01_dense/StatefulPartitionedCall2D
 02_dense/StatefulPartitionedCall 02_dense/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�	
�
$__inference_signature_wrapper_867282
input_2
unknown:(
	unknown_0:(
	unknown_1:((
	unknown_2:(
	unknown_3:(
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_867038o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
!__inference__wrapped_model_867038
input_2A
/model_2_01_dense_matmul_readvariableop_resource:(>
0model_2_01_dense_biasadd_readvariableop_resource:(A
/model_2_02_dense_matmul_readvariableop_resource:((>
0model_2_02_dense_biasadd_readvariableop_resource:(?
-model_2_output_matmul_readvariableop_resource:(<
.model_2_output_biasadd_readvariableop_resource:
identity

identity_1��'model_2/01_dense/BiasAdd/ReadVariableOp�&model_2/01_dense/MatMul/ReadVariableOp�'model_2/02_dense/BiasAdd/ReadVariableOp�&model_2/02_dense/MatMul/ReadVariableOp�%model_2/output/BiasAdd/ReadVariableOp�$model_2/output/MatMul/ReadVariableOp�
&model_2/01_dense/MatMul/ReadVariableOpReadVariableOp/model_2_01_dense_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_2/01_dense/MatMulMatMulinput_2.model_2/01_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
'model_2/01_dense/BiasAdd/ReadVariableOpReadVariableOp0model_2_01_dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_2/01_dense/BiasAddBiasAdd!model_2/01_dense/MatMul:product:0/model_2/01_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
model_2/01_dense/TanhTanh!model_2/01_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
&model_2/02_dense/MatMul/ReadVariableOpReadVariableOp/model_2_02_dense_matmul_readvariableop_resource*
_output_shapes

:((*
dtype0�
model_2/02_dense/MatMulMatMulmodel_2/01_dense/Tanh:y:0.model_2/02_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
'model_2/02_dense/BiasAdd/ReadVariableOpReadVariableOp0model_2_02_dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_2/02_dense/BiasAddBiasAdd!model_2/02_dense/MatMul:product:0/model_2/02_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
model_2/02_dense/TanhTanh!model_2/02_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
$model_2/output/MatMul/ReadVariableOpReadVariableOp-model_2_output_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_2/output/MatMulMatMulmodel_2/02_dense/Tanh:y:0,model_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%model_2/output/BiasAdd/ReadVariableOpReadVariableOp.model_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/output/BiasAddBiasAddmodel_2/output/MatMul:product:0-model_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitymodel_2/02_dense/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������(p

Identity_1Identitymodel_2/output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_2/01_dense/BiasAdd/ReadVariableOp'^model_2/01_dense/MatMul/ReadVariableOp(^model_2/02_dense/BiasAdd/ReadVariableOp'^model_2/02_dense/MatMul/ReadVariableOp&^model_2/output/BiasAdd/ReadVariableOp%^model_2/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2R
'model_2/01_dense/BiasAdd/ReadVariableOp'model_2/01_dense/BiasAdd/ReadVariableOp2P
&model_2/01_dense/MatMul/ReadVariableOp&model_2/01_dense/MatMul/ReadVariableOp2R
'model_2/02_dense/BiasAdd/ReadVariableOp'model_2/02_dense/BiasAdd/ReadVariableOp2P
&model_2/02_dense/MatMul/ReadVariableOp&model_2/02_dense/MatMul/ReadVariableOp2N
%model_2/output/BiasAdd/ReadVariableOp%model_2/output/BiasAdd/ReadVariableOp2L
$model_2/output/MatMul/ReadVariableOp$model_2/output/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�

�
(__inference_model_2_layer_call_fn_867320

inputs
unknown:(
	unknown_0:(
	unknown_1:((
	unknown_2:(
	unknown_3:(
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������(:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_867185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_20
serving_default_input_2:0���������<
02_dense0
StatefulPartitionedCall:0���������(:
output0
StatefulPartitionedCall:1���������tensorflow/serving/predict:�f
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
J
0
1
2
3
#4
$5"
trackable_list_wrapper
J
0
1
2
3
#4
$5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
*trace_0
+trace_1
,trace_2
-trace_32�
(__inference_model_2_layer_call_fn_867114
(__inference_model_2_layer_call_fn_867301
(__inference_model_2_layer_call_fn_867320
(__inference_model_2_layer_call_fn_867221�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z*trace_0z+trace_1z,trace_2z-trace_3
�
.trace_0
/trace_1
0trace_2
1trace_32�
C__inference_model_2_layer_call_and_return_conditional_losses_867345
C__inference_model_2_layer_call_and_return_conditional_losses_867370
C__inference_model_2_layer_call_and_return_conditional_losses_867241
C__inference_model_2_layer_call_and_return_conditional_losses_867261�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z.trace_0z/trace_1z0trace_2z1trace_3
�B�
!__inference__wrapped_model_867038input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
2serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
8trace_02�
)__inference_01_dense_layer_call_fn_867379�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z8trace_0
�
9trace_02�
D__inference_01_dense_layer_call_and_return_conditional_losses_867390�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z9trace_0
!:(201_dense/kernel
:(201_dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
?trace_02�
)__inference_02_dense_layer_call_fn_867399�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?trace_0
�
@trace_02�
D__inference_02_dense_layer_call_and_return_conditional_losses_867410�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
!:((202_dense/kernel
:(202_dense/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
Ftrace_02�
'__inference_output_layer_call_fn_867419�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0
�
Gtrace_02�
B__inference_output_layer_call_and_return_conditional_losses_867429�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
:(2output/kernel
:2output/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_2_layer_call_fn_867114input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_model_2_layer_call_fn_867301inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_model_2_layer_call_fn_867320inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_model_2_layer_call_fn_867221input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_2_layer_call_and_return_conditional_losses_867345inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_2_layer_call_and_return_conditional_losses_867370inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_2_layer_call_and_return_conditional_losses_867241input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_model_2_layer_call_and_return_conditional_losses_867261input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_867282input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_01_dense_layer_call_fn_867379inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_01_dense_layer_call_and_return_conditional_losses_867390inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_02_dense_layer_call_fn_867399inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_02_dense_layer_call_and_return_conditional_losses_867410inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_output_layer_call_fn_867419inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_output_layer_call_and_return_conditional_losses_867429inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
D__inference_01_dense_layer_call_and_return_conditional_losses_867390\/�,
%�"
 �
inputs���������
� "%�"
�
0���������(
� |
)__inference_01_dense_layer_call_fn_867379O/�,
%�"
 �
inputs���������
� "����������(�
D__inference_02_dense_layer_call_and_return_conditional_losses_867410\/�,
%�"
 �
inputs���������(
� "%�"
�
0���������(
� |
)__inference_02_dense_layer_call_fn_867399O/�,
%�"
 �
inputs���������(
� "����������(�
!__inference__wrapped_model_867038�#$0�-
&�#
!�
input_2���������
� "_�\
.
02_dense"�
02_dense���������(
*
output �
output����������
C__inference_model_2_layer_call_and_return_conditional_losses_867241�#$8�5
.�+
!�
input_2���������
p 

 
� "K�H
A�>
�
0/0���������(
�
0/1���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_867261�#$8�5
.�+
!�
input_2���������
p

 
� "K�H
A�>
�
0/0���������(
�
0/1���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_867345�#$7�4
-�*
 �
inputs���������
p 

 
� "K�H
A�>
�
0/0���������(
�
0/1���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_867370�#$7�4
-�*
 �
inputs���������
p

 
� "K�H
A�>
�
0/0���������(
�
0/1���������
� �
(__inference_model_2_layer_call_fn_867114�#$8�5
.�+
!�
input_2���������
p 

 
� "=�:
�
0���������(
�
1����������
(__inference_model_2_layer_call_fn_867221�#$8�5
.�+
!�
input_2���������
p

 
� "=�:
�
0���������(
�
1����������
(__inference_model_2_layer_call_fn_867301�#$7�4
-�*
 �
inputs���������
p 

 
� "=�:
�
0���������(
�
1����������
(__inference_model_2_layer_call_fn_867320�#$7�4
-�*
 �
inputs���������
p

 
� "=�:
�
0���������(
�
1����������
B__inference_output_layer_call_and_return_conditional_losses_867429\#$/�,
%�"
 �
inputs���������(
� "%�"
�
0���������
� z
'__inference_output_layer_call_fn_867419O#$/�,
%�"
 �
inputs���������(
� "�����������
$__inference_signature_wrapper_867282�#$;�8
� 
1�.
,
input_2!�
input_2���������"_�\
.
02_dense"�
02_dense���������(
*
output �
output���������
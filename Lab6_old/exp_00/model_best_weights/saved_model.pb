�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.0-dev202305182v1.12.1-94215-g4d38daffb968́
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
�
Adam/v/wyjsciowa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/wyjsciowa/bias
{
)Adam/v/wyjsciowa/bias/Read/ReadVariableOpReadVariableOpAdam/v/wyjsciowa/bias*
_output_shapes
:*
dtype0
�
Adam/m/wyjsciowa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/wyjsciowa/bias
{
)Adam/m/wyjsciowa/bias/Read/ReadVariableOpReadVariableOpAdam/m/wyjsciowa/bias*
_output_shapes
:*
dtype0
�
Adam/v/wyjsciowa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:N*(
shared_nameAdam/v/wyjsciowa/kernel
�
+Adam/v/wyjsciowa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/wyjsciowa/kernel*
_output_shapes

:N*
dtype0
�
Adam/m/wyjsciowa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:N*(
shared_nameAdam/m/wyjsciowa/kernel
�
+Adam/m/wyjsciowa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/wyjsciowa/kernel*
_output_shapes

:N*
dtype0
~
Adam/v/ukryta3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*$
shared_nameAdam/v/ukryta3/bias
w
'Adam/v/ukryta3/bias/Read/ReadVariableOpReadVariableOpAdam/v/ukryta3/bias*
_output_shapes
:N*
dtype0
~
Adam/m/ukryta3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*$
shared_nameAdam/m/ukryta3/bias
w
'Adam/m/ukryta3/bias/Read/ReadVariableOpReadVariableOpAdam/m/ukryta3/bias*
_output_shapes
:N*
dtype0
�
Adam/v/ukryta3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0N*&
shared_nameAdam/v/ukryta3/kernel

)Adam/v/ukryta3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/ukryta3/kernel*
_output_shapes

:0N*
dtype0
�
Adam/m/ukryta3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0N*&
shared_nameAdam/m/ukryta3/kernel

)Adam/m/ukryta3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/ukryta3/kernel*
_output_shapes

:0N*
dtype0
~
Adam/v/ukryta2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/v/ukryta2/bias
w
'Adam/v/ukryta2/bias/Read/ReadVariableOpReadVariableOpAdam/v/ukryta2/bias*
_output_shapes
:0*
dtype0
~
Adam/m/ukryta2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/m/ukryta2/bias
w
'Adam/m/ukryta2/bias/Read/ReadVariableOpReadVariableOpAdam/m/ukryta2/bias*
_output_shapes
:0*
dtype0
�
Adam/v/ukryta2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z0*&
shared_nameAdam/v/ukryta2/kernel

)Adam/v/ukryta2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/ukryta2/kernel*
_output_shapes

:Z0*
dtype0
�
Adam/m/ukryta2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z0*&
shared_nameAdam/m/ukryta2/kernel

)Adam/m/ukryta2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/ukryta2/kernel*
_output_shapes

:Z0*
dtype0
~
Adam/v/ukryta1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*$
shared_nameAdam/v/ukryta1/bias
w
'Adam/v/ukryta1/bias/Read/ReadVariableOpReadVariableOpAdam/v/ukryta1/bias*
_output_shapes
:Z*
dtype0
~
Adam/m/ukryta1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*$
shared_nameAdam/m/ukryta1/bias
w
'Adam/m/ukryta1/bias/Read/ReadVariableOpReadVariableOpAdam/m/ukryta1/bias*
_output_shapes
:Z*
dtype0
�
Adam/v/ukryta1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	Z*&
shared_nameAdam/v/ukryta1/kernel

)Adam/v/ukryta1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/ukryta1/kernel*
_output_shapes

:	Z*
dtype0
�
Adam/m/ukryta1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	Z*&
shared_nameAdam/m/ukryta1/kernel

)Adam/m/ukryta1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/ukryta1/kernel*
_output_shapes

:	Z*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
wyjsciowa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namewyjsciowa/bias
m
"wyjsciowa/bias/Read/ReadVariableOpReadVariableOpwyjsciowa/bias*
_output_shapes
:*
dtype0
|
wyjsciowa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:N*!
shared_namewyjsciowa/kernel
u
$wyjsciowa/kernel/Read/ReadVariableOpReadVariableOpwyjsciowa/kernel*
_output_shapes

:N*
dtype0
p
ukryta3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*
shared_nameukryta3/bias
i
 ukryta3/bias/Read/ReadVariableOpReadVariableOpukryta3/bias*
_output_shapes
:N*
dtype0
x
ukryta3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0N*
shared_nameukryta3/kernel
q
"ukryta3/kernel/Read/ReadVariableOpReadVariableOpukryta3/kernel*
_output_shapes

:0N*
dtype0
p
ukryta2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameukryta2/bias
i
 ukryta2/bias/Read/ReadVariableOpReadVariableOpukryta2/bias*
_output_shapes
:0*
dtype0
x
ukryta2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z0*
shared_nameukryta2/kernel
q
"ukryta2/kernel/Read/ReadVariableOpReadVariableOpukryta2/kernel*
_output_shapes

:Z0*
dtype0
p
ukryta1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameukryta1/bias
i
 ukryta1/bias/Read/ReadVariableOpReadVariableOpukryta1/bias*
_output_shapes
:Z*
dtype0
x
ukryta1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	Z*
shared_nameukryta1/kernel
q
"ukryta1/kernel/Read/ReadVariableOpReadVariableOpukryta1/kernel*
_output_shapes

:	Z*
dtype0
�
serving_default_ukryta1_inputPlaceholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_ukryta1_inputukryta1/kernelukryta1/biasukryta2/kernelukryta2/biasukryta3/kernelukryta3/biaswyjsciowa/kernelwyjsciowa/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_550847

NoOpNoOp
�6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�6
value�6B�6 B�6
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
<
0
1
2
3
$4
%5
,6
-7*
<
0
1
2
3
$4
%5
,6
-7*
* 
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

3trace_0
4trace_1* 

5trace_0
6trace_1* 
* 
�
7
_variables
8_iterations
9_learning_rate
:_index_dict
;
_momentums
<_velocities
=_update_step_xla*

>serving_default* 

0
1*

0
1*
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 
^X
VARIABLE_VALUEukryta1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEukryta1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
^X
VARIABLE_VALUEukryta2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEukryta2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
^X
VARIABLE_VALUEukryta3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEukryta3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
`Z
VARIABLE_VALUEwyjsciowa/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEwyjsciowa/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

[0
\1
]2*
* 
* 
* 
* 
* 
* 
�
80
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12
j13
k14
l15
m16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
^0
`1
b2
d3
f4
h5
j6
l7*
<
_0
a1
c2
e3
g4
i5
k6
m7*
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
8
n	variables
o	keras_api
	ptotal
	qcount*
H
r	variables
s	keras_api
	ttotal
	ucount
v
_fn_kwargs*
H
w	variables
x	keras_api
	ytotal
	zcount
{
_fn_kwargs*
`Z
VARIABLE_VALUEAdam/m/ukryta1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/ukryta1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/ukryta1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/ukryta1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/ukryta2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/ukryta2/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/ukryta2/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/ukryta2/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/ukryta3/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/ukryta3/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/ukryta3/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/ukryta3/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/wyjsciowa/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/wyjsciowa/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/wyjsciowa/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/wyjsciowa/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

n	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

t0
u1*

r	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

y0
z1*

w	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameukryta1/kernelukryta1/biasukryta2/kernelukryta2/biasukryta3/kernelukryta3/biaswyjsciowa/kernelwyjsciowa/bias	iterationlearning_rateAdam/m/ukryta1/kernelAdam/v/ukryta1/kernelAdam/m/ukryta1/biasAdam/v/ukryta1/biasAdam/m/ukryta2/kernelAdam/v/ukryta2/kernelAdam/m/ukryta2/biasAdam/v/ukryta2/biasAdam/m/ukryta3/kernelAdam/v/ukryta3/kernelAdam/m/ukryta3/biasAdam/v/ukryta3/biasAdam/m/wyjsciowa/kernelAdam/v/wyjsciowa/kernelAdam/m/wyjsciowa/biasAdam/v/wyjsciowa/biastotal_2count_2total_1count_1totalcountConst*-
Tin&
$2"*
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
__inference__traced_save_551140
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameukryta1/kernelukryta1/biasukryta2/kernelukryta2/biasukryta3/kernelukryta3/biaswyjsciowa/kernelwyjsciowa/bias	iterationlearning_rateAdam/m/ukryta1/kernelAdam/v/ukryta1/kernelAdam/m/ukryta1/biasAdam/v/ukryta1/biasAdam/m/ukryta2/kernelAdam/v/ukryta2/kernelAdam/m/ukryta2/biasAdam/v/ukryta2/biasAdam/m/ukryta3/kernelAdam/v/ukryta3/kernelAdam/m/ukryta3/biasAdam/v/ukryta3/biasAdam/m/wyjsciowa/kernelAdam/v/wyjsciowa/kernelAdam/m/wyjsciowa/biasAdam/v/wyjsciowa/biastotal_2count_2total_1count_1totalcount*,
Tin%
#2!*
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
"__inference__traced_restore_551245��
�
�
F__inference_sequential_layer_call_and_return_conditional_losses_550721
ukryta1_input 
ukryta1_550668:	Z
ukryta1_550670:Z 
ukryta2_550684:Z0
ukryta2_550686:0 
ukryta3_550700:0N
ukryta3_550702:N"
wyjsciowa_550715:N
wyjsciowa_550717:
identity��ukryta1/StatefulPartitionedCall�ukryta2/StatefulPartitionedCall�ukryta3/StatefulPartitionedCall�!wyjsciowa/StatefulPartitionedCall�
ukryta1/StatefulPartitionedCallStatefulPartitionedCallukryta1_inputukryta1_550668ukryta1_550670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta1_layer_call_and_return_conditional_losses_550667�
ukryta2/StatefulPartitionedCallStatefulPartitionedCall(ukryta1/StatefulPartitionedCall:output:0ukryta2_550684ukryta2_550686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta2_layer_call_and_return_conditional_losses_550683�
ukryta3/StatefulPartitionedCallStatefulPartitionedCall(ukryta2/StatefulPartitionedCall:output:0ukryta3_550700ukryta3_550702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta3_layer_call_and_return_conditional_losses_550699�
!wyjsciowa/StatefulPartitionedCallStatefulPartitionedCall(ukryta3/StatefulPartitionedCall:output:0wyjsciowa_550715wyjsciowa_550717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550714y
IdentityIdentity*wyjsciowa/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^ukryta1/StatefulPartitionedCall ^ukryta2/StatefulPartitionedCall ^ukryta3/StatefulPartitionedCall"^wyjsciowa/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2B
ukryta1/StatefulPartitionedCallukryta1/StatefulPartitionedCall2B
ukryta2/StatefulPartitionedCallukryta2/StatefulPartitionedCall2B
ukryta3/StatefulPartitionedCallukryta3/StatefulPartitionedCall2F
!wyjsciowa/StatefulPartitionedCall!wyjsciowa/StatefulPartitionedCall:&"
 
_user_specified_name550717:&"
 
_user_specified_name550715:&"
 
_user_specified_name550702:&"
 
_user_specified_name550700:&"
 
_user_specified_name550686:&"
 
_user_specified_name550684:&"
 
_user_specified_name550670:&"
 
_user_specified_name550668:V R
'
_output_shapes
:���������	
'
_user_specified_nameukryta1_input
�,
�
!__inference__wrapped_model_550654
ukryta1_inputC
1sequential_ukryta1_matmul_readvariableop_resource:	Z@
2sequential_ukryta1_biasadd_readvariableop_resource:ZC
1sequential_ukryta2_matmul_readvariableop_resource:Z0@
2sequential_ukryta2_biasadd_readvariableop_resource:0C
1sequential_ukryta3_matmul_readvariableop_resource:0N@
2sequential_ukryta3_biasadd_readvariableop_resource:NE
3sequential_wyjsciowa_matmul_readvariableop_resource:NB
4sequential_wyjsciowa_biasadd_readvariableop_resource:
identity��)sequential/ukryta1/BiasAdd/ReadVariableOp�(sequential/ukryta1/MatMul/ReadVariableOp�)sequential/ukryta2/BiasAdd/ReadVariableOp�(sequential/ukryta2/MatMul/ReadVariableOp�)sequential/ukryta3/BiasAdd/ReadVariableOp�(sequential/ukryta3/MatMul/ReadVariableOp�+sequential/wyjsciowa/BiasAdd/ReadVariableOp�*sequential/wyjsciowa/MatMul/ReadVariableOp�
(sequential/ukryta1/MatMul/ReadVariableOpReadVariableOp1sequential_ukryta1_matmul_readvariableop_resource*
_output_shapes

:	Z*
dtype0�
sequential/ukryta1/MatMulMatMulukryta1_input0sequential/ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Z�
)sequential/ukryta1/BiasAdd/ReadVariableOpReadVariableOp2sequential_ukryta1_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0�
sequential/ukryta1/BiasAddBiasAdd#sequential/ukryta1/MatMul:product:01sequential/ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Zv
sequential/ukryta1/ReluRelu#sequential/ukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:���������Z�
(sequential/ukryta2/MatMul/ReadVariableOpReadVariableOp1sequential_ukryta2_matmul_readvariableop_resource*
_output_shapes

:Z0*
dtype0�
sequential/ukryta2/MatMulMatMul%sequential/ukryta1/Relu:activations:00sequential/ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0�
)sequential/ukryta2/BiasAdd/ReadVariableOpReadVariableOp2sequential_ukryta2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
sequential/ukryta2/BiasAddBiasAdd#sequential/ukryta2/MatMul:product:01sequential/ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0v
sequential/ukryta2/ReluRelu#sequential/ukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:���������0�
(sequential/ukryta3/MatMul/ReadVariableOpReadVariableOp1sequential_ukryta3_matmul_readvariableop_resource*
_output_shapes

:0N*
dtype0�
sequential/ukryta3/MatMulMatMul%sequential/ukryta2/Relu:activations:00sequential/ukryta3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N�
)sequential/ukryta3/BiasAdd/ReadVariableOpReadVariableOp2sequential_ukryta3_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype0�
sequential/ukryta3/BiasAddBiasAdd#sequential/ukryta3/MatMul:product:01sequential/ukryta3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Nv
sequential/ukryta3/ReluRelu#sequential/ukryta3/BiasAdd:output:0*
T0*'
_output_shapes
:���������N�
*sequential/wyjsciowa/MatMul/ReadVariableOpReadVariableOp3sequential_wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:N*
dtype0�
sequential/wyjsciowa/MatMulMatMul%sequential/ukryta3/Relu:activations:02sequential/wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential/wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp4sequential_wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/wyjsciowa/BiasAddBiasAdd%sequential/wyjsciowa/MatMul:product:03sequential/wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%sequential/wyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp*^sequential/ukryta1/BiasAdd/ReadVariableOp)^sequential/ukryta1/MatMul/ReadVariableOp*^sequential/ukryta2/BiasAdd/ReadVariableOp)^sequential/ukryta2/MatMul/ReadVariableOp*^sequential/ukryta3/BiasAdd/ReadVariableOp)^sequential/ukryta3/MatMul/ReadVariableOp,^sequential/wyjsciowa/BiasAdd/ReadVariableOp+^sequential/wyjsciowa/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2V
)sequential/ukryta1/BiasAdd/ReadVariableOp)sequential/ukryta1/BiasAdd/ReadVariableOp2T
(sequential/ukryta1/MatMul/ReadVariableOp(sequential/ukryta1/MatMul/ReadVariableOp2V
)sequential/ukryta2/BiasAdd/ReadVariableOp)sequential/ukryta2/BiasAdd/ReadVariableOp2T
(sequential/ukryta2/MatMul/ReadVariableOp(sequential/ukryta2/MatMul/ReadVariableOp2V
)sequential/ukryta3/BiasAdd/ReadVariableOp)sequential/ukryta3/BiasAdd/ReadVariableOp2T
(sequential/ukryta3/MatMul/ReadVariableOp(sequential/ukryta3/MatMul/ReadVariableOp2Z
+sequential/wyjsciowa/BiasAdd/ReadVariableOp+sequential/wyjsciowa/BiasAdd/ReadVariableOp2X
*sequential/wyjsciowa/MatMul/ReadVariableOp*sequential/wyjsciowa/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
'
_output_shapes
:���������	
'
_user_specified_nameukryta1_input
�
�
*__inference_wyjsciowa_layer_call_fn_550916

inputs
unknown:N
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������N: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550912:&"
 
_user_specified_name550910:O K
'
_output_shapes
:���������N
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_550787
ukryta1_input
unknown:	Z
	unknown_0:Z
	unknown_1:Z0
	unknown_2:0
	unknown_3:0N
	unknown_4:N
	unknown_5:N
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallukryta1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_550745o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550783:&"
 
_user_specified_name550781:&"
 
_user_specified_name550779:&"
 
_user_specified_name550777:&"
 
_user_specified_name550775:&"
 
_user_specified_name550773:&"
 
_user_specified_name550771:&"
 
_user_specified_name550769:V R
'
_output_shapes
:���������	
'
_user_specified_nameukryta1_input
�

�
C__inference_ukryta1_layer_call_and_return_conditional_losses_550867

inputs0
matmul_readvariableop_resource:	Z-
biasadd_readvariableop_resource:Z
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Zr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������ZP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������Za
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������ZS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
C__inference_ukryta3_layer_call_and_return_conditional_losses_550907

inputs0
matmul_readvariableop_resource:0N-
biasadd_readvariableop_resource:N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Nr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:N*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������NP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������Na
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������NS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
(__inference_ukryta2_layer_call_fn_550876

inputs
unknown:Z0
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta2_layer_call_and_return_conditional_losses_550683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������0<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550872:&"
 
_user_specified_name550870:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�

�
C__inference_ukryta1_layer_call_and_return_conditional_losses_550667

inputs0
matmul_readvariableop_resource:	Z-
biasadd_readvariableop_resource:Z
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Zr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������ZP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������Za
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������ZS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
(__inference_ukryta1_layer_call_fn_550856

inputs
unknown:	Z
	unknown_0:Z
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta1_layer_call_and_return_conditional_losses_550667o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550852:&"
 
_user_specified_name550850:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
C__inference_ukryta3_layer_call_and_return_conditional_losses_550699

inputs0
matmul_readvariableop_resource:0N-
biasadd_readvariableop_resource:N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Nr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:N*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������NP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������Na
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������NS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_550766
ukryta1_input
unknown:	Z
	unknown_0:Z
	unknown_1:Z0
	unknown_2:0
	unknown_3:0N
	unknown_4:N
	unknown_5:N
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallukryta1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_550721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550762:&"
 
_user_specified_name550760:&"
 
_user_specified_name550758:&"
 
_user_specified_name550756:&"
 
_user_specified_name550754:&"
 
_user_specified_name550752:&"
 
_user_specified_name550750:&"
 
_user_specified_name550748:V R
'
_output_shapes
:���������	
'
_user_specified_nameukryta1_input
�

�
C__inference_ukryta2_layer_call_and_return_conditional_losses_550887

inputs0
matmul_readvariableop_resource:Z0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������0a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������0S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�	
�
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550926

inputs0
matmul_readvariableop_resource:N-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������N: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������N
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_551245
file_prefix1
assignvariableop_ukryta1_kernel:	Z-
assignvariableop_1_ukryta1_bias:Z3
!assignvariableop_2_ukryta2_kernel:Z0-
assignvariableop_3_ukryta2_bias:03
!assignvariableop_4_ukryta3_kernel:0N-
assignvariableop_5_ukryta3_bias:N5
#assignvariableop_6_wyjsciowa_kernel:N/
!assignvariableop_7_wyjsciowa_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: ;
)assignvariableop_10_adam_m_ukryta1_kernel:	Z;
)assignvariableop_11_adam_v_ukryta1_kernel:	Z5
'assignvariableop_12_adam_m_ukryta1_bias:Z5
'assignvariableop_13_adam_v_ukryta1_bias:Z;
)assignvariableop_14_adam_m_ukryta2_kernel:Z0;
)assignvariableop_15_adam_v_ukryta2_kernel:Z05
'assignvariableop_16_adam_m_ukryta2_bias:05
'assignvariableop_17_adam_v_ukryta2_bias:0;
)assignvariableop_18_adam_m_ukryta3_kernel:0N;
)assignvariableop_19_adam_v_ukryta3_kernel:0N5
'assignvariableop_20_adam_m_ukryta3_bias:N5
'assignvariableop_21_adam_v_ukryta3_bias:N=
+assignvariableop_22_adam_m_wyjsciowa_kernel:N=
+assignvariableop_23_adam_v_wyjsciowa_kernel:N7
)assignvariableop_24_adam_m_wyjsciowa_bias:7
)assignvariableop_25_adam_v_wyjsciowa_bias:%
assignvariableop_26_total_2: %
assignvariableop_27_count_2: %
assignvariableop_28_total_1: %
assignvariableop_29_count_1: #
assignvariableop_30_total: #
assignvariableop_31_count: 
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_ukryta1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_ukryta1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_ukryta2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_ukryta2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_ukryta3_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_ukryta3_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_wyjsciowa_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_wyjsciowa_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_adam_m_ukryta1_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_v_ukryta1_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_m_ukryta1_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_v_ukryta1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_m_ukryta2_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_v_ukryta2_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_m_ukryta2_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_v_ukryta2_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_m_ukryta3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_v_ukryta3_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_m_ukryta3_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_v_ukryta3_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_m_wyjsciowa_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_v_wyjsciowa_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_wyjsciowa_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_wyjsciowa_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_2Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_2Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_1Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_1Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_33IdentityIdentity_32:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_33Identity_33:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:% !

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_2:'#
!
_user_specified_name	total_2:51
/
_user_specified_nameAdam/v/wyjsciowa/bias:51
/
_user_specified_nameAdam/m/wyjsciowa/bias:73
1
_user_specified_nameAdam/v/wyjsciowa/kernel:73
1
_user_specified_nameAdam/m/wyjsciowa/kernel:3/
-
_user_specified_nameAdam/v/ukryta3/bias:3/
-
_user_specified_nameAdam/m/ukryta3/bias:51
/
_user_specified_nameAdam/v/ukryta3/kernel:51
/
_user_specified_nameAdam/m/ukryta3/kernel:3/
-
_user_specified_nameAdam/v/ukryta2/bias:3/
-
_user_specified_nameAdam/m/ukryta2/bias:51
/
_user_specified_nameAdam/v/ukryta2/kernel:51
/
_user_specified_nameAdam/m/ukryta2/kernel:3/
-
_user_specified_nameAdam/v/ukryta1/bias:3/
-
_user_specified_nameAdam/m/ukryta1/bias:51
/
_user_specified_nameAdam/v/ukryta1/kernel:51
/
_user_specified_nameAdam/m/ukryta1/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:.*
(
_user_specified_namewyjsciowa/bias:0,
*
_user_specified_namewyjsciowa/kernel:,(
&
_user_specified_nameukryta3/bias:.*
(
_user_specified_nameukryta3/kernel:,(
&
_user_specified_nameukryta2/bias:.*
(
_user_specified_nameukryta2/kernel:,(
&
_user_specified_nameukryta1/bias:.*
(
_user_specified_nameukryta1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
$__inference_signature_wrapper_550847
ukryta1_input
unknown:	Z
	unknown_0:Z
	unknown_1:Z0
	unknown_2:0
	unknown_3:0N
	unknown_4:N
	unknown_5:N
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallukryta1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_550654o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550843:&"
 
_user_specified_name550841:&"
 
_user_specified_name550839:&"
 
_user_specified_name550837:&"
 
_user_specified_name550835:&"
 
_user_specified_name550833:&"
 
_user_specified_name550831:&"
 
_user_specified_name550829:V R
'
_output_shapes
:���������	
'
_user_specified_nameukryta1_input
�
�
F__inference_sequential_layer_call_and_return_conditional_losses_550745
ukryta1_input 
ukryta1_550724:	Z
ukryta1_550726:Z 
ukryta2_550729:Z0
ukryta2_550731:0 
ukryta3_550734:0N
ukryta3_550736:N"
wyjsciowa_550739:N
wyjsciowa_550741:
identity��ukryta1/StatefulPartitionedCall�ukryta2/StatefulPartitionedCall�ukryta3/StatefulPartitionedCall�!wyjsciowa/StatefulPartitionedCall�
ukryta1/StatefulPartitionedCallStatefulPartitionedCallukryta1_inputukryta1_550724ukryta1_550726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta1_layer_call_and_return_conditional_losses_550667�
ukryta2/StatefulPartitionedCallStatefulPartitionedCall(ukryta1/StatefulPartitionedCall:output:0ukryta2_550729ukryta2_550731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta2_layer_call_and_return_conditional_losses_550683�
ukryta3/StatefulPartitionedCallStatefulPartitionedCall(ukryta2/StatefulPartitionedCall:output:0ukryta3_550734ukryta3_550736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta3_layer_call_and_return_conditional_losses_550699�
!wyjsciowa/StatefulPartitionedCallStatefulPartitionedCall(ukryta3/StatefulPartitionedCall:output:0wyjsciowa_550739wyjsciowa_550741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550714y
IdentityIdentity*wyjsciowa/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^ukryta1/StatefulPartitionedCall ^ukryta2/StatefulPartitionedCall ^ukryta3/StatefulPartitionedCall"^wyjsciowa/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������	: : : : : : : : 2B
ukryta1/StatefulPartitionedCallukryta1/StatefulPartitionedCall2B
ukryta2/StatefulPartitionedCallukryta2/StatefulPartitionedCall2B
ukryta3/StatefulPartitionedCallukryta3/StatefulPartitionedCall2F
!wyjsciowa/StatefulPartitionedCall!wyjsciowa/StatefulPartitionedCall:&"
 
_user_specified_name550741:&"
 
_user_specified_name550739:&"
 
_user_specified_name550736:&"
 
_user_specified_name550734:&"
 
_user_specified_name550731:&"
 
_user_specified_name550729:&"
 
_user_specified_name550726:&"
 
_user_specified_name550724:V R
'
_output_shapes
:���������	
'
_user_specified_nameukryta1_input
�
�
(__inference_ukryta3_layer_call_fn_550896

inputs
unknown:0N
	unknown_0:N
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ukryta3_layer_call_and_return_conditional_losses_550699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������N<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name550892:&"
 
_user_specified_name550890:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�	
�
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550714

inputs0
matmul_readvariableop_resource:N-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������N: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������N
 
_user_specified_nameinputs
��
�
__inference__traced_save_551140
file_prefix7
%read_disablecopyonread_ukryta1_kernel:	Z3
%read_1_disablecopyonread_ukryta1_bias:Z9
'read_2_disablecopyonread_ukryta2_kernel:Z03
%read_3_disablecopyonread_ukryta2_bias:09
'read_4_disablecopyonread_ukryta3_kernel:0N3
%read_5_disablecopyonread_ukryta3_bias:N;
)read_6_disablecopyonread_wyjsciowa_kernel:N5
'read_7_disablecopyonread_wyjsciowa_bias:,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: A
/read_10_disablecopyonread_adam_m_ukryta1_kernel:	ZA
/read_11_disablecopyonread_adam_v_ukryta1_kernel:	Z;
-read_12_disablecopyonread_adam_m_ukryta1_bias:Z;
-read_13_disablecopyonread_adam_v_ukryta1_bias:ZA
/read_14_disablecopyonread_adam_m_ukryta2_kernel:Z0A
/read_15_disablecopyonread_adam_v_ukryta2_kernel:Z0;
-read_16_disablecopyonread_adam_m_ukryta2_bias:0;
-read_17_disablecopyonread_adam_v_ukryta2_bias:0A
/read_18_disablecopyonread_adam_m_ukryta3_kernel:0NA
/read_19_disablecopyonread_adam_v_ukryta3_kernel:0N;
-read_20_disablecopyonread_adam_m_ukryta3_bias:N;
-read_21_disablecopyonread_adam_v_ukryta3_bias:NC
1read_22_disablecopyonread_adam_m_wyjsciowa_kernel:NC
1read_23_disablecopyonread_adam_v_wyjsciowa_kernel:N=
/read_24_disablecopyonread_adam_m_wyjsciowa_bias:=
/read_25_disablecopyonread_adam_v_wyjsciowa_bias:+
!read_26_disablecopyonread_total_2: +
!read_27_disablecopyonread_count_2: +
!read_28_disablecopyonread_total_1: +
!read_29_disablecopyonread_count_1: )
read_30_disablecopyonread_total: )
read_31_disablecopyonread_count: 
savev2_const
identity_65��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_ukryta1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_ukryta1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	Z*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	Za

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:	Zy
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_ukryta1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_ukryta1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:Z*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Z_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:Z{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_ukryta2_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_ukryta2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Z0*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Z0c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:Z0y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_ukryta2_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_ukryta2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:0{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_ukryta3_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_ukryta3_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0N*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0Nc

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:0Ny
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_ukryta3_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_ukryta3_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:N*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Na
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:N}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_wyjsciowa_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_wyjsciowa_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:N*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Ne
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:N{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_wyjsciowa_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_wyjsciowa_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead/read_10_disablecopyonread_adam_m_ukryta1_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp/read_10_disablecopyonread_adam_m_ukryta1_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	Z*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	Ze
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:	Z�
Read_11/DisableCopyOnReadDisableCopyOnRead/read_11_disablecopyonread_adam_v_ukryta1_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp/read_11_disablecopyonread_adam_v_ukryta1_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	Z*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	Ze
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:	Z�
Read_12/DisableCopyOnReadDisableCopyOnRead-read_12_disablecopyonread_adam_m_ukryta1_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp-read_12_disablecopyonread_adam_m_ukryta1_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:Z*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Za
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:Z�
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_adam_v_ukryta1_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_adam_v_ukryta1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:Z*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Za
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:Z�
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_adam_m_ukryta2_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_adam_m_ukryta2_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Z0*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Z0e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:Z0�
Read_15/DisableCopyOnReadDisableCopyOnRead/read_15_disablecopyonread_adam_v_ukryta2_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp/read_15_disablecopyonread_adam_v_ukryta2_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Z0*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Z0e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:Z0�
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_adam_m_ukryta2_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_adam_m_ukryta2_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_adam_v_ukryta2_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp-read_17_disablecopyonread_adam_v_ukryta2_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:0�
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_adam_m_ukryta3_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_adam_m_ukryta3_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0N*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0Ne
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:0N�
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_v_ukryta3_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_v_ukryta3_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0N*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0Ne
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:0N�
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_adam_m_ukryta3_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_adam_m_ukryta3_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:N*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Na
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:N�
Read_21/DisableCopyOnReadDisableCopyOnRead-read_21_disablecopyonread_adam_v_ukryta3_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp-read_21_disablecopyonread_adam_v_ukryta3_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:N*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Na
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:N�
Read_22/DisableCopyOnReadDisableCopyOnRead1read_22_disablecopyonread_adam_m_wyjsciowa_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp1read_22_disablecopyonread_adam_m_wyjsciowa_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:N*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Ne
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:N�
Read_23/DisableCopyOnReadDisableCopyOnRead1read_23_disablecopyonread_adam_v_wyjsciowa_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp1read_23_disablecopyonread_adam_v_wyjsciowa_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:N*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Ne
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:N�
Read_24/DisableCopyOnReadDisableCopyOnRead/read_24_disablecopyonread_adam_m_wyjsciowa_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp/read_24_disablecopyonread_adam_m_wyjsciowa_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_adam_v_wyjsciowa_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_adam_v_wyjsciowa_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_2^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_2^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_28/DisableCopyOnReadDisableCopyOnRead!read_28_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp!read_28_disablecopyonread_total_1^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_29/DisableCopyOnReadDisableCopyOnRead!read_29_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp!read_29_disablecopyonread_count_1^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_30/DisableCopyOnReadDisableCopyOnReadread_30_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpread_30_disablecopyonread_total^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_31/DisableCopyOnReadDisableCopyOnReadread_31_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpread_31_disablecopyonread_count^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 */
dtypes%
#2!	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_64Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_65IdentityIdentity_64:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_65Identity_65:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=!9

_output_shapes
: 

_user_specified_nameConst:% !

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_2:'#
!
_user_specified_name	total_2:51
/
_user_specified_nameAdam/v/wyjsciowa/bias:51
/
_user_specified_nameAdam/m/wyjsciowa/bias:73
1
_user_specified_nameAdam/v/wyjsciowa/kernel:73
1
_user_specified_nameAdam/m/wyjsciowa/kernel:3/
-
_user_specified_nameAdam/v/ukryta3/bias:3/
-
_user_specified_nameAdam/m/ukryta3/bias:51
/
_user_specified_nameAdam/v/ukryta3/kernel:51
/
_user_specified_nameAdam/m/ukryta3/kernel:3/
-
_user_specified_nameAdam/v/ukryta2/bias:3/
-
_user_specified_nameAdam/m/ukryta2/bias:51
/
_user_specified_nameAdam/v/ukryta2/kernel:51
/
_user_specified_nameAdam/m/ukryta2/kernel:3/
-
_user_specified_nameAdam/v/ukryta1/bias:3/
-
_user_specified_nameAdam/m/ukryta1/bias:51
/
_user_specified_nameAdam/v/ukryta1/kernel:51
/
_user_specified_nameAdam/m/ukryta1/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:.*
(
_user_specified_namewyjsciowa/bias:0,
*
_user_specified_namewyjsciowa/kernel:,(
&
_user_specified_nameukryta3/bias:.*
(
_user_specified_nameukryta3/kernel:,(
&
_user_specified_nameukryta2/bias:.*
(
_user_specified_nameukryta2/kernel:,(
&
_user_specified_nameukryta1/bias:.*
(
_user_specified_nameukryta1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
C__inference_ukryta2_layer_call_and_return_conditional_losses_550683

inputs0
matmul_readvariableop_resource:Z0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������0a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������0S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
ukryta1_input6
serving_default_ukryta1_input:0���������	=
	wyjsciowa0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�t
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
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
�
3trace_0
4trace_12�
+__inference_sequential_layer_call_fn_550766
+__inference_sequential_layer_call_fn_550787�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3trace_0z4trace_1
�
5trace_0
6trace_12�
F__inference_sequential_layer_call_and_return_conditional_losses_550721
F__inference_sequential_layer_call_and_return_conditional_losses_550745�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z5trace_0z6trace_1
�B�
!__inference__wrapped_model_550654ukryta1_input"�
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
�
7
_variables
8_iterations
9_learning_rate
:_index_dict
;
_momentums
<_velocities
=_update_step_xla"
experimentalOptimizer
,
>serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Dtrace_02�
(__inference_ukryta1_layer_call_fn_550856�
���
FullArgSpec
args�

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
annotations� *
 zDtrace_0
�
Etrace_02�
C__inference_ukryta1_layer_call_and_return_conditional_losses_550867�
���
FullArgSpec
args�

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
annotations� *
 zEtrace_0
 :	Z2ukryta1/kernel
:Z2ukryta1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ktrace_02�
(__inference_ukryta2_layer_call_fn_550876�
���
FullArgSpec
args�

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
annotations� *
 zKtrace_0
�
Ltrace_02�
C__inference_ukryta2_layer_call_and_return_conditional_losses_550887�
���
FullArgSpec
args�

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
annotations� *
 zLtrace_0
 :Z02ukryta2/kernel
:02ukryta2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_02�
(__inference_ukryta3_layer_call_fn_550896�
���
FullArgSpec
args�

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
annotations� *
 zRtrace_0
�
Strace_02�
C__inference_ukryta3_layer_call_and_return_conditional_losses_550907�
���
FullArgSpec
args�

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
annotations� *
 zStrace_0
 :0N2ukryta3/kernel
:N2ukryta3/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
Ytrace_02�
*__inference_wyjsciowa_layer_call_fn_550916�
���
FullArgSpec
args�

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
annotations� *
 zYtrace_0
�
Ztrace_02�
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550926�
���
FullArgSpec
args�

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
annotations� *
 zZtrace_0
": N2wyjsciowa/kernel
:2wyjsciowa/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
[0
\1
]2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_sequential_layer_call_fn_550766ukryta1_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_550787ukryta1_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_550721ukryta1_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_550745ukryta1_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
80
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12
j13
k14
l15
m16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
^0
`1
b2
d3
f4
h5
j6
l7"
trackable_list_wrapper
X
_0
a1
c2
e3
g4
i5
k6
m7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_550847ukryta1_input"�
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
(__inference_ukryta1_layer_call_fn_550856inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_ukryta1_layer_call_and_return_conditional_losses_550867inputs"�
���
FullArgSpec
args�

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
(__inference_ukryta2_layer_call_fn_550876inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_ukryta2_layer_call_and_return_conditional_losses_550887inputs"�
���
FullArgSpec
args�

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
(__inference_ukryta3_layer_call_fn_550896inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_ukryta3_layer_call_and_return_conditional_losses_550907inputs"�
���
FullArgSpec
args�

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
*__inference_wyjsciowa_layer_call_fn_550916inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550926inputs"�
���
FullArgSpec
args�

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
annotations� *
 
N
n	variables
o	keras_api
	ptotal
	qcount"
_tf_keras_metric
^
r	variables
s	keras_api
	ttotal
	ucount
v
_fn_kwargs"
_tf_keras_metric
^
w	variables
x	keras_api
	ytotal
	zcount
{
_fn_kwargs"
_tf_keras_metric
%:#	Z2Adam/m/ukryta1/kernel
%:#	Z2Adam/v/ukryta1/kernel
:Z2Adam/m/ukryta1/bias
:Z2Adam/v/ukryta1/bias
%:#Z02Adam/m/ukryta2/kernel
%:#Z02Adam/v/ukryta2/kernel
:02Adam/m/ukryta2/bias
:02Adam/v/ukryta2/bias
%:#0N2Adam/m/ukryta3/kernel
%:#0N2Adam/v/ukryta3/kernel
:N2Adam/m/ukryta3/bias
:N2Adam/v/ukryta3/bias
':%N2Adam/m/wyjsciowa/kernel
':%N2Adam/v/wyjsciowa/kernel
!:2Adam/m/wyjsciowa/bias
!:2Adam/v/wyjsciowa/bias
.
p0
q1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
.
t0
u1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_550654y$%,-6�3
,�)
'�$
ukryta1_input���������	
� "5�2
0
	wyjsciowa#� 
	wyjsciowa����������
F__inference_sequential_layer_call_and_return_conditional_losses_550721x$%,->�;
4�1
'�$
ukryta1_input���������	
p

 
� ",�)
"�
tensor_0���������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_550745x$%,->�;
4�1
'�$
ukryta1_input���������	
p 

 
� ",�)
"�
tensor_0���������
� �
+__inference_sequential_layer_call_fn_550766m$%,->�;
4�1
'�$
ukryta1_input���������	
p

 
� "!�
unknown����������
+__inference_sequential_layer_call_fn_550787m$%,->�;
4�1
'�$
ukryta1_input���������	
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_550847�$%,-G�D
� 
=�:
8
ukryta1_input'�$
ukryta1_input���������	"5�2
0
	wyjsciowa#� 
	wyjsciowa����������
C__inference_ukryta1_layer_call_and_return_conditional_losses_550867c/�,
%�"
 �
inputs���������	
� ",�)
"�
tensor_0���������Z
� �
(__inference_ukryta1_layer_call_fn_550856X/�,
%�"
 �
inputs���������	
� "!�
unknown���������Z�
C__inference_ukryta2_layer_call_and_return_conditional_losses_550887c/�,
%�"
 �
inputs���������Z
� ",�)
"�
tensor_0���������0
� �
(__inference_ukryta2_layer_call_fn_550876X/�,
%�"
 �
inputs���������Z
� "!�
unknown���������0�
C__inference_ukryta3_layer_call_and_return_conditional_losses_550907c$%/�,
%�"
 �
inputs���������0
� ",�)
"�
tensor_0���������N
� �
(__inference_ukryta3_layer_call_fn_550896X$%/�,
%�"
 �
inputs���������0
� "!�
unknown���������N�
E__inference_wyjsciowa_layer_call_and_return_conditional_losses_550926c,-/�,
%�"
 �
inputs���������N
� ",�)
"�
tensor_0���������
� �
*__inference_wyjsciowa_layer_call_fn_550916X,-/�,
%�"
 �
inputs���������N
� "!�
unknown���������
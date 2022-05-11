# shell本身是解释执行的，不需要编译，内嵌的解析器解析
# 系统命令会去根目录下面的bin目录查找
# date-查看日期
# cd-转路径
# ls-罗列目录
# echo $SHELL-查看shell版本
# (cd ..;ls)-加了小括号不改变路径
# vi .bashrc-查看根目录隐藏文件bashrc，保存退出时用:wq，不保存则q:
# . .bashrc运行当前目录bashrc
# env查看环境变量 env | grep $PATH
# liunx系统不以后缀名作为区分依据，所以扩展名不是sh也可以

#shell只有字符串类型
#环境变量和本地变量，变量定义VAR=10视为字符串，不能有空格
#控制语句
#函数
#面向对象（不支持）

#！/usr/bin/zsh

echo "Hello World!"
/bin/pwd
/bin/ls

# cd
VAR=10
env grep | VAR
env grep | $VAR
# export 把本地变量转环境变量
# 删除环境变量unset
# *当前全部,如*.sh 代替字母?，[]匹配中括号范围内一个 如1-9 a-z
ls *.sh
ls ????.sh
ls [a-z]???.sh
# $取值符号
VAR=date
echo VAR
echo $VAR #对变量名取值，建议${}
unset VAR
cd
VAR=$(date) #对命令取结果
echo $VAR
# 反引号·是取值
echo `date` #对命令取结果
VAR=45
echo $($VAR+3) #只能用+-*/()运算符，[]会将里面运算符作为运算
#echo $[2#10+5] #2进制10和5求和




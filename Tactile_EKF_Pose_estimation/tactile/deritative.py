from sympy import *

# x = Symbol("x")
# y = diff(x**3+x, x)
# print(y)
# result = y.subs('x', 1)
# print(result)

x, y = symbols('x, y')

z = x**2+y**2+x*y+2
print("z:", z)
result = z.subs({x: 1, y: 2})   # 用数值分别对x、y进行替换
print("result:", result)

dx = diff(z, x)   # 对x求偏导
print("dx:", dx)
result = dx.subs({x: 1, y: 2})
print("result:", result)

dy = diff(z, y)   # 对y求偏导
print("dy:", dy)
result = dy.subs({x: 1, y: 2})
print("result:", result)


# subs函数可以将算式中的符号进行替换，它有3种调用方式：
# expression.subs(x, y) : 将算式中的x替换成y
# expression.subs({x:y,u:v}) : 使用字典进行多次替换
# expression.subs([(x,y),(u,v)]) : 使用列表进行多次替换

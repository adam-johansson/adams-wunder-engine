
import sys
sys.path.append("./../../")


from CCE.src.components import compressor



p_in = 1e5
T_in = 272.15 + 20

p1 = 1.8e5
p2 = 2.6e5
p3 = 3.75e5
p4 = 4.55e5

eta = 0.9

ps = [p1,p2,p3,p4]

for p in ps:
    pr = p/p_in
    T_out = compressor(T_in, p_in, 1, eta, pr)
    print(f"T_out: {T_out} K")


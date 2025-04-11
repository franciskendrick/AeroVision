landmaarks = ["class"]
for val in range(1, 33+1):
    landmaarks += [f"x{val}", f"y{val}", f"z{val}", f"v{val}"]

x = ",".join(landmaarks)
print(x)
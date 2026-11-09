import math

class RPR:
    def __init__(self, L1, L2, rail_min, rail_max):
        self.rail_min = rail_min
        self.rail_max = rail_max
        self.L1 = L1
        self.L2 = L2

    def inv_kinematic(self, x, y_s):
        rail_d = y_s
        y = 0

        if rail_d < self.rail_min:
            rail_d = self.rail_min
        elif rail_d > self.rail_max:
            rail_d = self.rail_max
        else:
            pass

        if x < 12:
            return rail_d, 90, 90

        D = (pow(x, 2) + pow(y, 2) - pow(self.L1, 2) - pow(self.L2, 2)) / (2 * self.L1 * self.L2)
        if D < -1 or D > 1:
            print("unreachable")
        else:
            # elbow
            e_angel1 = math.acos(D)

            im_var = self.L1 + self.L2 * math.cos(e_angel1)
            in_var = self.L2 * math.sin(e_angel1)

            s_angel1 = math.atan2(y, x) - math.atan2(in_var, im_var)
            s_angel1 = math.degrees(s_angel1)
            e_angel1 = math.degrees(e_angel1)

            e_angel2 = -math.acos(D)

            im_var = self.L1 + self.L2 * math.cos(e_angel2)
            in_var = self.L2 * math.sin(e_angel2)

            s_angel2 = math.atan2(y, x) - math.atan2(in_var, im_var)
            s_angel2 = math.degrees(s_angel2)
            e_angel2 = math.degrees(e_angel2)


            if s_angel1 < 0 and s_angel2 < 0:
                s = 0
            elif s_angel1 < 0:
                s = s_angel2
                e = e_angel2
            else:
                s = s_angel1
                e = e_angel1

            return rail_d, 180 + e, s


if __name__ == "__main__":
    L1 = 20
    L2 = 10
    rail_min = 0
    rail_max = 30

    rpr = RPR(L1, L2, rail_min, rail_max)
    rail, elbow, shoulder = rpr.inv_kinematic(30, 20)
    print("(rail, elbow, shoulder)", rail, int(elbow), int(shoulder))

    wrist = 270 - int(elbow) - int(shoulder)
    print(wrist)

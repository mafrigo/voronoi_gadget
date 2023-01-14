__all__ = ["generate_snapshot", "PseudoSnap", "orient_snap"]

import numpy as np
try:
    import pygad
    # import pygadmpa as mpa
    no_pygad = False
except ImportError:
    no_pygad = True
    print("Running without pygad")


def openfile(filename):
    return filename


def generate_snapshot(N_stars, size=1, velocity_scale=200.):
    snap = dict()
    np.random.seed(1)
    snap["ID"] = np.arange(N_stars)
    snap["mass"] = np.full(N_stars, 1.)
    snap["pos"] = np.random.choice([-1., 1.], (N_stars, 3)) * np.random.exponential(size, (N_stars, 3))
    snap["vel"] = np.full((N_stars, 3), velocity_scale) + np.random.normal(0., 2.*velocity_scale, (N_stars, 3))
    snap["vel"][:, 2] *= np.sign(snap["pos"][:, 1])
    snap["vel"][:, 1] *= np.sign(snap["pos"][:, 2])
    snap["vel"][:, 0] *= np.sign(snap["pos"][:, 2])
    snap["pos"] += np.random.normal(0., 0.5*size, (N_stars, 3))
    print(np.min(abs(snap["pos"])), np.max(abs(snap["pos"])))
    print(np.min(snap["vel"]), np.min(abs(snap["vel"])), np.max(abs(snap["vel"])))
    return snap


def orient_snap(snap, axisorientation=1, rangmom=1., gal_R200=0.1, ensure_rotdir=False):
    """
    Orients snap to the principal axes of inertia.
    axisorientation==1: intermediate axis is along the line-of-sight (edge-on orientation).
    axisorientation==2: major axis is along the line-of-sight.
    axisorientation==3: minor axis is along the line-of-sight (face-on orientation).

    -rangmom is the radius used to estimate the direction of the angular
      momentum
    -The galaxy is also cut at gal_R200 times the virial radius.
    """
    s, halo, gal = pygad.prepare_zoom(snap, star_form=None, gas_trace=None, mode='ssc',
                                      to_physical=False)
    if axisorientation == 0:
        return s
    posx = np.copy(s["pos"][:, 0])
    posy = np.copy(s["pos"][:, 1])
    posz = np.copy(s["pos"][:, 2])
    velx = np.copy(s["vel"][:, 0])
    vely = np.copy(s["vel"][:, 1])
    velz = np.copy(s["vel"][:, 2])
    if axisorientation == 1:
        s["pos"][:, 2] = posy
        s["pos"][:, 1] = posx
        s["pos"][:, 0] = posz
        s["vel"][:, 2] = vely
        s["vel"][:, 1] = velx
        s["vel"][:, 0] = velz
    elif axisorientation == 2:
        s["pos"][:, 2] = posx
        s["pos"][:, 0] = posz
        s["vel"][:, 2] = velx
        s["vel"][:, 0] = velz
    elif axisorientation == 3:
        s["pos"][:, 0] = posy
        s["pos"][:, 1] = posx
        s["vel"][:, 0] = vely
        s["vel"][:, 1] = velx
    else:
        raise IOError("axisorientation must be 0, 1, 2 or 3")
    # Orient galaxy so that the angular momentum vector has
    # a consistent direction:
    R200, M200 = pygad.analysis.virial_info(s)
    s = s[pygad.BallMask(gal_R200 * R200)]

    if ensure_rotdir:
        # ensures that the angular momentum vector is always pointing in the same direction
        rangmom = 5.
        angmomy = snap["mass"] * (snap["vel"][:, 2] * snap["pos"][:, 1] - snap["vel"][:, 1] * snap["pos"][:, 2])
        if np.mean(angmomy[snap["pos"][:, 0] ** 2 + snap["pos"][:, 1] ** 2 < rangmom ** 2]) > 0:
            snap["pos"] = -snap["pos"]
    return s


class PseudoSnap(object):
    """
    For every particle in the original snapshot, creates npseudoparticles copies
    distributed according to a 3D gaussian around the original location, with
    sigma sigmapp. The object then contains the positions of the new
    pseudoparticles and the particle ID of their original particle.
    The function getproperty() can then return any other property of the
    pseudoparticles.
    """

    def __init__(self, snap, npseudoparticles, sigmapp, qty_spread):
        self.sigma = sigmapp
        self.npp = npseudoparticles
        self.qty_spread = qty_spread
        self.snap = snap
        self._descriptor = "pseudosnap"

        x = snap["pos"][:, 0]
        y = snap["pos"][:, 1]
        z = snap["pos"][:, 2]
        ids = snap["ID"]
        try:
            posunits = x.units
        except:
            posunits = None
        xg = np.ravel(np.random.normal(x.repeat(npseudoparticles), sigmapp,
                                       len(x) * npseudoparticles))
        yg = np.ravel(np.random.normal(y.repeat(npseudoparticles), sigmapp,
                                       len(x) * npseudoparticles))
        zg = np.ravel(np.random.normal(z.repeat(npseudoparticles), sigmapp,
                                       len(x) * npseudoparticles))

        idsg = np.array(ids.repeat(npseudoparticles))

        self._pos = np.transpose([xg, yg, zg])
        if posunits is not None:
            self._pos = pygad.UnitArr(self._pos, posunits)
        self._ids = idsg

    def get_snap(self):
        return self.snap

    def get_property(self, qtylabel):
        """
        Calculates a snapshot property for the pseudosnapshot and returns it.

        Note: relies on the fact that the order of particles in the original
        snapshot stays the same!
        TODO: A better version could use the ids in the pseudosnap to get the
        correct properties.
        """
        # ids=self._ids
        qty = self.snap[qtylabel]
        qtyg = self._expandqty(qty, qtylabel=qtylabel, sigmaqty=self.qty_spread)
        return qtyg

    def get(self, expr, units=None, namespace=None):
        qty = self.snap.get(expr, units=units, namespace=namespace)
        qtyg = self._expandqty(qty, qtylabel=expr, sigmaqty=self.qty_spread)
        return qtyg

    def __getitem__(self, key):
        if key == 'pos':
            return self._pos
        if key == 'ID':
            return self._ids
        if key not in ["pos", "ID"]:
            return self.get_property(key)

    def __getattr__(self, name):
        """
        Gets subsnaps from the pseudo-snapshot by recalculating it, which makes
        it very inefficient. Normally PseudoSnap should be used only after all
        snapshot transformations (orientation, subsnap selection, etc.) have been
        done.
        """
        if no_pygad:
            raise IOError("Subsnapshots of a PseudoSnap only work with pygad")
        import pygad
        if name in pygad.gadget.families:
            fam_snap = pygad.snapshot.FamilySubSnap(self.snap, name)
            fam_pseudosnap = PseudoSnap(fam_snap, self.npp, self.sigma)
            # setattr(self, name, fam_snap)
            return fam_pseudosnap
        else:
            raise IOError("Block " + name + "not available in current snapshot")

    def __len__(self):
        return len(self.snap) * self.npp

    def _expandqty(self, qty, qtylabel=None, sigmaqty=-1):
        try:
            qtyunits = qty.units
        except AttributeError:
            qtyunits = None
        if len(np.shape(qty)) > 1:  # 3D quantities
            qtyg = np.zeros((np.shape(qty)[0] * self.npp, np.shape(qty)[1]))
            for i in np.arange(np.shape(qty)[1]):
                if sigmaqty > 0:
                    qtyg[:, i] = np.ravel(np.random.normal(qty[:, i].repeat(self.npp),
                                                           sigmaqty,
                                                           len(qty[:, i]) * self.npp))
                else:
                    qtyg[:, i] = np.array(qty[:, i].repeat(self.npp))
        else:  # 1D quantities
            if sigmaqty > 0:
                qtyg = np.ravel(np.random.normal(qty.repeat(self.npp), sigmaqty,
                                                 len(qty) * self.npp))
            else:
                qtyg = np.array(qty.repeat(self.npp))
        if qtylabel in ["mass", "lum", "lum_u", "lum_b", "lum_v", "lum_r",
                        "lum_k", "momentum", "angmom", "Ekin", "Epot", "E",
                        "jcirc", "jzjc", "LX"]:  # (integral quantities)
            qtyg = qtyg / self.npp
        if qtyunits is None:
            return qtyg
        else:
            return pygad.UnitArr(qtyg, qtyunits)
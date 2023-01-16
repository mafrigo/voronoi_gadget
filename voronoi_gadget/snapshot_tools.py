__all__ = ["generate_snapshot", "PseudoSnap", "orient_snap", "openfile", "rotate"]

import numpy as np
try:
    import pygadmpa as pygad
    no_pygad = False
except ImportError:
    no_pygad = True
    print("Running without pygad")


def openfile(snap, subsnap="stars", force_orient=False, ensure_rotdir=False, angle=0.):
    if type(snap) == str:
        a, b, snap = pygad.prepare_zoom(snap, mode='ssc') # Opens simulation, centers on stellar component, cuts at 0.1*virial radius
    if subsnap == "stars":
        snap = snap.stars
    if subsnap == "gas":
        snap = snap.gas
    if force_orient:
        snap = orient_snap(snap, ensure_rotdir=ensure_rotdir, inclination=angle)
    return snap


def generate_snapshot(N_stars, size=1, velocity_scale=200.):
    snap = dict()
    np.random.seed(1)
    snap["ID"] = np.arange(N_stars)
    snap["mass"] = np.full(N_stars, 1.)
    snap["pos"] = np.random.choice([-1., 1.], (N_stars, 3)) * np.random.exponential(size, (N_stars, 3)) # spherical system
    snap["pos"][:, 1] /= 1.2 # intermediate axis should be slightly smaller
    snap["pos"][:, 2] /= 3. # minor axis much smaller -> make system flattened
    snap["vel"] = np.full((N_stars, 3), 0.1)
    vel = np.random.normal(2.*velocity_scale, 0.1*velocity_scale, N_stars)
    rad = np.sqrt(snap["pos"][:, 0]**2 + snap["pos"][:, 1]**2)
    snap["vel"][:, 0] = - vel * (snap["pos"][:, 1]/rad)
    snap["vel"][:, 1] = vel * (snap["pos"][:, 0]/rad) # velocity oriented tangentially to position
    snap["pos"] += np.random.normal(0., 0.5 * size, (N_stars, 3)) # extra randomness
    print(np.min(abs(snap["pos"])), np.max(abs(snap["pos"])))
    print(np.min(snap["vel"]), np.min(abs(snap["vel"])), np.max(abs(snap["vel"])))
    return snap


def rotate(qty, inclination, z_rotation):
    """
    Rotate given vector (qty) according to given angles (inclination and rotation around z-axis)
    """
    inc = inclination * np.pi / 180.
    phi = z_rotation * np.pi / 180.
    qtyx = qty[0] * np.cos(phi) - qty[1] * np.sin(phi)
    qtyy = (qty[0] * np.sin(phi) + qty[1] * np.cos(phi)) * np.cos(inc) - qty[2] * np.sin(inc)
    qtyz = (qty[0] * np.sin(phi) + qty[1] * np.cos(phi)) * np.sin(inc) + qty[2] * np.cos(inc)
    return [qtyx, qtyy, qtyz]


def orient_snap(snap, inclination=0., z_rotation=0., ensure_rotdir=False, include_extra_qtys=[]):
    """
    Orients snap to the principal axes of inertia (coordinate 0 = major axis, coordinate 1 = intermediate axis,
    coordinate 2 = minor axis).
    If inclination and or z_rotation are given, rotates the galaxy by those angles; inclination is the rotation around
    the major axis, while z_rotation around the minor axis.
    If ensure_rotdir is true, the galaxy will always have the angular momantum vector pointing down.
    """
    snap["vg_inclination"] = inclination
    snap["vg_z_rotation"] = z_rotation

    if inclination != 0. or z_rotation != 0.:
        posx = np.copy(snap["pos"][:, 0])
        posy = np.copy(snap["pos"][:, 1])
        posz = np.copy(snap["pos"][:, 2])
        velx = np.copy(snap["vel"][:, 0])
        vely = np.copy(snap["vel"][:, 1])
        velz = np.copy(snap["vel"][:, 2])
        newpos = rotate([posx,posy,posz], inclination, z_rotation)
        snap["pos"][:, 0] = newpos[0]
        snap["pos"][:, 1] = newpos[1]
        snap["pos"][:, 2] = newpos[2]
        newvel = rotate([velx,vely,velz], inclination, z_rotation)
        snap["vel"][:, 0] = newvel[0]
        snap["vel"][:, 1] = newvel[1]
        snap["vel"][:, 2] = newvel[2]

    if ensure_rotdir:
        # ensures that the angular momentum vector is always pointing in the same direction
        rangmom = 2.
        angmomy = snap["mass"] * (snap["vel"][:, 2] * snap["pos"][:, 1] - snap["vel"][:, 1] * snap["pos"][:, 2])
        if np.mean(angmomy[snap["pos"][:, 0] ** 2 + snap["pos"][:, 1] ** 2 < rangmom ** 2]) > 0:
            snap["pos"] = -snap["pos"]
    return snap


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
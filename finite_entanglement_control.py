import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from math import ceil
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.linalg import eigh, eigvalsh, helmert
from scipy.optimize import least_squares, linprog, root
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import eigsh, splu

BLOCKS = ((1, 2), (5, 6))
BASELINE = (1, 5)
TARGET = np.array([0.05, -0.05])
BOND_CAP = 0.15
MEAN_CAP = 0.149
FIELD_CAP = 0.30
DIFF_CAP = 0.05
GAP_FLOOR = 0.02
NUMERICAL_PAD = 2e-9
X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.diag([1., -1.])
IDENTITY = np.eye(2)
PILOT_PAIRS = {3801: (0, 5), 3802: (1, 6), 3803: (1, 6), 3804: (1, 6)}
HELD_OUT = tuple(range(3805, 3817))
FAMILIES = ["XXZ", "Ising", "Ising-symmetric", "XY", "XY-separated"]
GEOMETRIES = ["adjacent blocks", "interleaved pairs", "mixed geometry"]
AXES = {"A+", "A-", "B+", "B-"}


def pauli_product(n, xmask=0, zmask=0):
    j = np.arange(2**n)
    bits = np.zeros(len(j), dtype=int)
    for i in range(n):
        if (zmask >> i) & 1:
            bits ^= (j >> i) & 1
    return csr_matrix((1 - 2*bits, (j ^ xmask, j)), shape=(len(j),)*2)


class Chain:
    def __init__(self, family, n):
        self.n, self.family = n, family
        self.states = np.arange(2**n)
        bits = (self.states[:,None] >> np.arange(n)) & 1
        self.sectors = [np.flatnonzero(bits.sum(axis=1)==k) for k in range(n+1)]
        self.basis = self.sectors[n//2] if family=='XXZ' else self.states
        self.z = [pauli_product(n,zmask=1 << j) for j in range(n)]
        self.x = [pauli_product(n,xmask=1 << j) for j in range(n)]
        self.zz = [pauli_product(n,zmask=3 << j) for j in range(n-1)]
        if family == 'XXZ':
            self.fields = [op-self.z[-1] for op in self.z[:-1]]
            bonds = [pauli_product(n,xmask=3<<j) @ (eye(2**n)-zz) + .8*zz
                     for j,zz in enumerate(self.zz)]
            observables = self.z[:-1]
        elif family=='Ising-symmetric':
            self.fields=self.x
            bonds=self.zz
            observables=self.x
        elif family in ('XY','XY-separated'):
            self.fields = self.z
            yy = [-pauli_product(n,xmask=3<<j,zmask=3<<j) for j in range(n-1)]
            xx = [self.x[j]@self.x[j+1] for j in range(n-1)]
            bonds = [u+.42*v for u,v in zip(xx,yy)] if family=='XY' else xx+yy
            observables = self.z
        else:
            self.fields = self.z + self.x
            bonds = self.zz
            observables = self.fields
        self.ops = self.fields + bonds
        self.nb = len(bonds)
        self.nf = len(self.fields)
        self.control = np.array([a[self.basis][:,self.basis].toarray() for a in self.ops])
        self.local = np.array([a[self.basis][:,self.basis].toarray() for a in observables])
        audit_indices = self.basis
        self.audit = np.array([a[audit_indices][:,audit_indices].toarray() for a in
                               (self.z if family in ('XXZ','XY','XY-separated') else self.z+self.x)])
        self.dim = len(self.basis)
        self._build_reduction_masks()

    def _build_reduction_masks(self):
        self.maps = {}
        for A in combinations(range(self.n),2):
            complement = [j for j in range(self.n) if j not in A]
            row = sum(((self.basis>>j)&1)<<k for k,j in enumerate(A))
            col = sum(((self.basis>>j)&1)<<k for k,j in enumerate(complement))
            self.maps[A] = row,col

    def state(self, par):
        return eigh(np.tensordot(par,self.control,axes=1), check_finite=False, driver='evd')

    def params(self, seed):
        rng = np.random.default_rng(seed)
        if self.family=='XXZ':
            return np.r_[rng.normal(0,.18,self.n-1),rng.uniform(.85,1.15,self.nb)]
        if self.family=='Ising-symmetric':
            return -np.r_[rng.uniform(.95,1.2,self.n),rng.uniform(.85,1.15,self.nb)]
        if self.family=='XY':
            return np.r_[rng.uniform(.70,1.0,self.n),rng.uniform(.8,1.15,self.nb)]
        if self.family=='XY-separated':
            return np.r_[rng.uniform(1.15,1.30,self.n),rng.uniform(.85,1.15,self.n-1),rng.uniform(.35,.48,self.n-1)]
        return np.r_[rng.uniform(.18,.52,self.n),rng.uniform(-1.2,-.8,self.n),rng.uniform(-1.05,-.65,self.nb)]

    def analyze(self, par, blocks):
        w,v = self.state(par)
        psi = v[:,0]
        applied = self.control@psi
        dv = -((applied@v[:,1:])/(w[1:]-w[0])) @ v[:,1:].T
        local = (self.local@psi)@psi
        dc = 2*(self.local@psi)@dv.T
        s,ds = [],[]
        for block in blocks:
            row,col = self.maps[tuple(sorted(block))]
            array = np.zeros((4,2**(self.n-2)))
            array[row,col] = psi
            rho = array@array.T
            p = np.sum(rho*rho)
            s.append(-np.log2(p))
            ds.append(-4*(rho@array)[row,col]@dv.T/(p*np.log(2)))
        return np.r_[local,s],np.vstack((dc,ds)),psi,(self.audit@psi)@psi,w[:2]

    def full_gap(self, par, levels):
        if self.family!='XXZ':
            return levels[1]-levels[0],0.
        h=sum((x*a for x,a in zip(par,self.ops)),csr_matrix(self.ops[0].shape))
        differences = []
        for k,b in enumerate(self.sectors):
            if k==self.n//2:
                continue
            w=eigh(h[b][:,b].toarray(),eigvals_only=True,subset_by_index=[0,0])[0]
            differences.append((self.n-2*k,w-levels[0]))
        slopes, offsets=np.array(differences).T
        fit=linprog([0,-1],A_ub=np.c_[-slopes,np.ones(len(slopes))],b_ub=offsets,
                    bounds=[(None,None),(None,levels[1]-levels[0])],method='highs')
        if not fit.success:
            raise ValueError(fit.message)
        return fit.x[1],fit.x[0]

    def reduced(self, jac, bond_indices=None):
        if bond_indices is None:
            bond_indices=np.arange(self.nb)
        bond_indices=np.asarray(bond_indices)
        dc, ge=jac[:-2],jac[-2:]
        fields=np.linalg.solve(dc[:,:self.nf],-dc[:,self.nf+bond_indices])
        directions=np.vstack((fields,np.eye(self.nb)[:,bond_indices]))
        return ge@directions, directions


def block_configs(n):
    return [((0,1),(n-2,n-1)), ((1,2),(n-3,n-2))]


def tensor_operator(matrix, site, n=8):
    left = np.eye(2**site)
    right = np.eye(2**(n-site-1))
    return np.kron(np.kron(left, matrix), right)


def xxz_bonds(n=8):
    pair = np.kron(X, X) + np.kron(Y, Y) + .8*np.kron(Z, Z)
    return np.asarray([
        np.kron(np.kron(np.eye(2**i), pair.real), np.eye(2**(n-i-2)))
        for i in range(n-1)
    ])


def reduced_state(psi, block, n=8):
    other = tuple(i for i in range(n) if i not in block)
    array = psi.reshape((2,)*n).transpose(block + other).reshape(2**len(block), -1)
    return array @ array.conj().T


def entropies(psi):
    states = [reduced_state(psi, block) for block in BLOCKS]
    return -np.log2([np.vdot(rho, rho).real for rho in states])


def zero_field_study():
    ops = xxz_bonds()
    h0 = ops.sum(axis=0)
    da, db = ops[list(BASELINE)]
    values = np.linspace(-BOND_CAP, BOND_CAP, 81)

    def solve(edits):
        energy, basis = eigh(h0 + edits[0]*da + edits[1]*db, subset_by_index=[0, 1],
                             check_finite=False)
        ground = basis[:, 0]
        return entropies(ground), energy[1]-energy[0], ground

    center, _, _ = solve([0., 0.])
    data = []
    for a in values:
        for b in values:
            outputs, gap, ground = solve([a, b])
            site_error = max(np.max(np.abs(reduced_state(ground, (i,)) - IDENTITY/2))
                             for i in range(8))
            data.append([a, b, *(outputs-center), gap, site_error])
    data = np.array(data)
    step = values[1]-values[0]
    norms = [np.max(np.abs(eigvalsh(op, check_finite=False))) for op in (da, db)]
    gap_min = float(data[:, 4].min())
    gap_bound = gap_min - sum(norms)*step - 1e-9
    allowances = 4*np.array(norms)*step/(2*gap_bound*np.log(2)) + 1e-9
    margins = []
    for k in range(2):
        others = 1-k
        lo = np.min(data[np.isclose(data[:, k], -BOND_CAP), 2+k])
        hi = -np.max(data[np.isclose(data[:, k], BOND_CAP), 2+k])
        margins.append([lo-allowances[others], hi-allowances[others]])

    common, differential = .149, .001
    diff_norm = np.max(np.abs(eigvalsh(da-db, check_finite=False)))
    extent = 8*diff_norm*differential/(gap_bound*np.log(2))
    step_fd = 1e-5
    response = np.column_stack([(solve(step_fd*axis)[0]-solve(-step_fd*axis)[0])/(2*step_fd)
                                for axis in np.eye(2)])
    transform = np.array([[1, 1], [1, -1]])
    singular = np.linalg.svd(response @ transform @ np.diag([common, differential]),
                             compute_uv=False)
    restricted = np.array([[x, y, *(solve(transform @ [x, y])[0]-center)]
                           for x in np.linspace(-common, common, 81)
                           for y in (-differential, 0., differential)])
    endpoint = root(lambda edits: solve(edits)[0]-center-TARGET, np.zeros(2), tol=1e-10)
    reached, endpoint_gap, _ = solve(endpoint.x)
    np.testing.assert_allclose(reached-center, TARGET, rtol=0, atol=1e-11)
    max_site_error = np.max(data[:, 5])
    if (np.min(margins) < .10 or max_site_error > 1e-10 or singular[-1] <= 1e-3
            or extent >= .1 or np.max(np.abs(endpoint.x)) >= BOND_CAP):
        raise RuntimeError("Symmetric control coverage or exclusion failed")
    np.testing.assert_allclose([gap_bound, np.min(margins), extent],
                               [1.16672266, .110286, .03957], atol=6e-6, rtol=0)

    return {
        "grid": 81, "bond_half_width": BOND_CAP, "square_half_width_bits": .10,
        "face_margins_bits": margins, "max_site_density_matrix_entry_error": max_site_error,
        "initial_entropies_bits": center.tolist(), "sampled_gap": gap_min,
        "full_gap_lower_bound": gap_bound, "restricted_difference_bound_bits": extent,
        "restricted_normalized_singular_values": singular.tolist(),
        "target_bits": TARGET.tolist(), "achieved_bits": (reached-center).tolist(),
        "endpoint_bond_edits": endpoint.x.tolist(), "endpoint_gap": endpoint_gap,
        "restricted_common_limit": common, "restricted_differential_limit": differential,
        "grid_data": pd.DataFrame(data, columns=["bond_a", "bond_b", "delta_a",
                                                    "delta_b", "gap", "local_error"]),
        "restricted_data": pd.DataFrame(restricted, columns=["common", "differential",
                                                              "delta_a", "delta_b"]),
    }


POLARIZED = Chain("XXZ", 8)
HELMERT = helmert(8, full=False).T
TRANSFORM = np.zeros((14, 14))
TRANSFORM[:7, :7] = HELMERT[:-1]
TRANSFORM[7:, 7:] = np.eye(7)
SECTOR_OPERATORS = np.einsum("ai,ajk->ijk", TRANSFORM, POLARIZED.control)
SPIN_OPERATORS = np.array([op.toarray() for op in POLARIZED.ops])
FIELD_NORM = np.sqrt(8.)


def polarized_state(p, selected=BLOCKS):
    v,j,psi,loc,e=POLARIZED.analyze(TRANSFORM@p,selected)
    return np.r_[HELMERT.T@loc,v[-2:]],np.vstack((HELMERT.T@(np.r_[np.eye(7),-np.ones((1,7))]@j[:-2]),j[-2:]))@TRANSFORM,psi,loc,e


def polarized_reference(seed):
    rng=np.random.default_rng(seed)
    return np.r_[HELMERT.T@rng.normal(0,.24,8),rng.uniform(.85,1.15,7)]


def compensated_state(initial, delta, ref, guess=None, pair=BASELINE):
    p=initial.copy()
    p[7 + np.asarray(pair)]+=delta
    def advance(u):
        p[:7]=u
        v,j,*_=polarized_state(p)
        return v[:7]-ref[:7],j[:7,:7]
    x=root(advance,p[:7] if guess is None else guess,jac=True,tol=1e-10)
    residual=np.max(abs(advance(x.x)[0]))
    if residual>1e-10:
        raise ValueError(residual)
    return p.copy()


def compensated_response(jacobian, pair=BASELINE):
    """Return the entropy Jacobian after eliminating the seven field coordinates."""
    field_to_constraint = jacobian[:7, :7]
    bond_to_constraint = jacobian[:7, 7 + np.asarray(pair)]
    return jacobian[-2:, 7 + np.asarray(pair)] - jacobian[-2:, :7] @ np.linalg.solve(
        field_to_constraint, bond_to_constraint
    )


def full_space_levels(p):
    h=np.einsum('i,ijk->jk',TRANSFORM@p,SPIN_OPERATORS)
    ans=[]
    for k,ids in enumerate(POLARIZED.sectors):
        ans.extend(eigvalsh(h[np.ix_(ids,ids)],subset_by_index=[0,min(len(ids)-1,1)],check_finite=False) if k==4
                   else eigvalsh(h[np.ix_(ids,ids)],subset_by_index=[0,0],check_finite=False))
    return np.sort(ans)[:2]


def certify_cell(p, G, first, centers, S0, center_constraint_residual, pair=BASELINE):
    axes_indices = 7 + np.asarray(pair)
    matrix=np.einsum('i,ijk->jk',p,SECTOR_OPERATORS)
    e,v=eigh(matrix,check_finite=False,driver='evd')
    psi=v[:,0]
    response=-np.linalg.solve(G[:7,:7],G[:7,axes_indices])
    dirs=np.zeros((2,len(p)))
    dirs[:,:7]=response.T
    dirs[np.arange(2),axes_indices]=1
    U=dirs@SECTOR_OPERATORS.reshape(len(p),-1)
    applied=U.reshape(2,len(psi),len(psi))@psi
    dl=applied@psi
    w=-((applied@v[:,1:])/(e[1:]-e[0]))@v[:,1:].T
    # Bounds over all signed combinations of the two cell half-widths.
    axes=np.asarray(centers)
    hnorm=0.
    for z in ((1,1),(1,-1)):
        field=HELMERT@(z[0]*axes[0]*dirs[0,:7]+z[1]*axes[1]*dirs[1,:7])
        hnorm=max(hnorm,np.abs(field).sum()+2.8*axes.sum())
    amp=sum(np.linalg.norm(w,axis=1)*axes)
    shifted=sum(np.abs(dl)*axes)+hnorm
    # (H+deltaH - (E+deltaE))(psi+psi') = (deltaH-deltaE)psi'.
    spectral=e[1]-e[0]-shifted
    assert spectral>0
    diff_mat=U.reshape(2,len(psi),len(psi))-dl[:,None,None]*np.eye(len(psi))
    a0,a1=axes
    rem=a0*a0*np.linalg.norm(diff_mat[0]@w[0])+a1*a1*np.linalg.norm(diff_mat[1]@w[1])
    rem+=a0*a1*np.linalg.norm(diff_mat[0]@w[1]+diff_mat[1]@w[0])
    eta=rem/spectral+NUMERICAL_PAD
    # First-order term cancels; account for trial-state normalization and alignment.
    residual=center_constraint_residual+2*FIELD_NORM*(amp**2+np.sqrt(2)*eta)+NUMERICAL_PAD
    V=(SECTOR_OPERATORS[:7]@psi)
    V=V-(V@psi)[:,None]*psi
    sv=np.linalg.svd(V,compute_uv=False)[-1]
    # Bound for the minimum susceptibility in a ball centered at the linear predictor.
    # Transfer the projected observables through the state change, then use the full
    # energy span in the conserved sector as an upper bound on every denominator.
    for rho in np.geomspace(1e-7,.12,95):
        distance=hnorm+FIELD_NORM*rho
        gap=e[1]-e[0]-2*distance
        if gap<=0.: break
        # At the predictor the normalized state shift is bounded via psi', residual
        # and phase alignment. From the predictor to any point in the ball, the
        # direct reduced-resolvent derivative integrates over a gapped line segment.
        state=amp+np.sqrt(2)*eta+FIELD_NORM*rho/gap+NUMERICAL_PAD
        lower=max(0.,sv-3*FIELD_NORM*state)
        strong=2*lower**2/(e[-1]-e[0]+2*distance)
        if rho*strong>residual:
            break
    else:
        return None
    if gap<=0 or not rho*strong>residual:
        return None
    fullgap=first[1]-first[0]-2*(hnorm+FIELD_NORM*rho)-NUMERICAL_PAD
    field=np.max(np.abs(HELMERT@(p[:7]-S0[:7]))+np.abs(HELMERT@response)@axes)+rho+NUMERICAL_PAD
    # Bound the face change of Renyi entropy from the center of this cell.
    # The compensating state differs by 'state' in Hilbert space. A geodesic of
    # normalized states has length 2 asin(state/2). |dS/darclength| <= 4 / ln 2.
    entropy=8*np.arcsin(min(1.,state/2))/np.log(2)+NUMERICAL_PAD
    return fullgap,field,entropy,rho,strong,amp


def polarized_box(seed,n,half):
    p0=polarized_reference(seed)
    start,_,_,m0,_=polarized_state(p0)
    if np.min(abs(m0))<.01:
        raise ValueError('The selected example requires each spin polarized by at least .01.')
    w=half/n
    grid=np.linspace(-half+w,half-w,n)
    worst_gap=float('inf')
    maxfield=maxroot=maxbloch=maxe=0.
    susceptibility=float('inf')
    min_sampled_gap=float('inf')
    max_sector_error=0.
    faces=np.full((2,2),float('inf'))
    count=0
    guess=None
    for i,a in enumerate(grid):
        for j in (range(n) if i%2==0 else reversed(range(n))):
            b=grid[j]
            p=compensated_state(p0,[a,b],start,guess=guess)
            guess=p[:7].copy()
            y,G,_,obs,energies=polarized_state(p)
            maxbloch=max(maxbloch,np.max(abs(obs-m0)))
            full_levels=full_space_levels(p)
            sector_error=abs(energies[0]-full_levels[0])
            assert sector_error<1e-10
            max_sector_error=max(max_sector_error,sector_error)
            min_sampled_gap=min(min_sampled_gap,full_levels[1]-energies[0])
            residual=np.linalg.norm(y[:7]-start[:7])
            out=certify_cell(p,G,full_levels,[w,w],p0,residual)
            if out is None:
                return dict(cells_verified=count,total=n**2)
            g,f,ent,rho,smin,amp=out
            worst_gap=min(worst_gap,g)
            maxfield=max(maxfield,f)
            maxroot=max(maxroot,rho)
            susceptibility=min(susceptibility,smin)
            maxe=max(maxe,ent)
            for k,ix in enumerate((i,j)):
                if ix==0:
                    faces[k,0]=min(faces[k,0],y[7+k]-start[7+k]-ent)
                if ix==n-1:
                    faces[k,1]=min(faces[k,1],start[7+k]-y[7+k]-ent)
            count+=1
    return dict(cells_verified=count,total=n**2,seed=seed,half_width=half,
                initial_bloch=list(m0),initial_bits=list(start[-2:]),
                min_gap=worst_gap,max_field_edit=maxfield,max_cell_root_radius=maxroot,
                min_susceptibility_lower_bound=susceptibility,max_grid_local_drift=maxbloch,
                max_entropy_cell_allowance=maxe,face_margins=faces.tolist(),
                min_sampled_full_space_gap=min_sampled_gap,
                max_ground_energy_sector_difference=max_sector_error,
                certified=bool(worst_gap>GAP_FLOOR and maxfield<FIELD_CAP and faces.min()>0.05))


EDIT_TRANSFORM = np.array([[1., 1.], [1., -1.]])


def refine_faces(seed=3801, half=.12, n=251):
    p0=polarized_reference(seed)
    vals,*_=polarized_state(p0)
    w=half/n
    grid=np.linspace(-half+w,half-w,n)
    margins=np.full((2,2),np.inf)
    gaps=[]
    fields=[]
    for k in (0,1):
        for face in (0,1):
            p=p0.copy()
            for other in grid:
                controls=np.zeros(2)
                controls[k]=grid[0 if face==0 else -1]
                controls[1-k]=other
                p=compensated_state(p0,controls,vals,guess=p[:7])
                v,j,_,_,e=polarized_state(p)
                full=full_space_levels(p)
                assert abs(full[0]-e[0])<1e-10
                residual=np.linalg.norm(v[:7]-vals[:7])
                bound=certify_cell(p,j,full,[w,w],p0,residual)
                assert bound is not None
                g,f,entropy,*_=bound
                margins[k,face]=min(margins[k,face],((-1)**face)*(v[7+k]-vals[7+k])-entropy)
                gaps.append(g)
                fields.append(f)
    return {'seed':seed,'half_width':half,'grid_edge_cells':4*n,'face_margins':margins.tolist(),
         'min_gap_edge':min(gaps),'max_field_edit_edge':max(fields)}


def restricted_box(seed=3802, nx=401, ny=3):
    p0=polarized_reference(seed)
    start,j,*_=polarized_state(p0)
    common,differential=MEAN_CAP,0.001
    widths=np.array([common/nx+differential/ny]*2)
    extent=0.
    maxfield=0.
    gap=np.inf
    guess=None
    for x in np.linspace(-common+common/nx,common-common/nx,nx):
        for y in np.linspace(-differential+differential/ny,differential-differential/ny,ny):
            p=compensated_state(p0,EDIT_TRANSFORM@[x,y],start,guess=guess)
            guess=p[:7].copy()
            k,J,_,_,levels=polarized_state(p)
            low=full_space_levels(p)
            assert abs(low[0]-levels[0])<1e-10
            residual=np.linalg.norm(k[:7]-start[:7])
            values=certify_cell(p,J,low,widths,p0,residual)
            if values is None:
                raise ValueError((x,y,'no local root certificate'))
            g,f,error,*_=values
            gap=min(g,gap)
            maxfield=max(maxfield,f)
            extent=max(extent,abs((k[-2]-k[-1])-(start[-2]-start[-1]))+2*error)
    matrix=compensated_response(j)@EDIT_TRANSFORM@np.diag([common,differential])
    sv=np.linalg.svd(matrix,compute_uv=False)
    ans={'seed':seed,'cells':nx*ny,'grid':[nx,ny],
         'common_limit':common,'differential_limit':differential,
         'entropy_difference_offset_upper_bound':extent,
         'minimum_gap':gap,'max_field_edit':maxfield,'normalized_singular_values':list(sv),
         'opposing_target_bits':list(TARGET),
         'excluded':bool(extent<abs(TARGET[0]-TARGET[1])
                         and gap>GAP_FLOOR and maxfield<FIELD_CAP)}
    return ans


def sample_offsets(initial, edits):
    start, *_ = polarized_state(initial)
    offsets = []
    guess = initial[:7]
    for change in edits:
        parameters = compensated_state(initial, change, start, guess=guess)
        guess = parameters[:7]
        offsets.append(polarized_state(parameters)[0][-2:] - start[-2:])
    return np.asarray(offsets)


def combine_coverage(interior, edge=None):
    if interior["cells_verified"] != interior["total"]:
        raise RuntimeError(f"Seed {interior['seed']} has uncertified control cells")
    gap = interior["min_gap"]
    field = interior["max_field_edit"]
    margins = interior["face_margins"]
    if edge is not None:
        if (interior["seed"], interior["half_width"]) != (edge["seed"], edge["half_width"]):
            raise ValueError("The interior and edge checks must cover the same control box")
        gap = min(gap, edge["min_gap_edge"])
        field = max(field, edge["max_field_edit_edge"])
        margins = edge["face_margins"]

    # Round in the conservative direction at five decimal places.
    half_width = np.floor(np.min(margins) * 1e5) / 1e5
    result = {
        "seed": interior["seed"],
        "independent_bond_edit_half_width": interior["half_width"],
        "certified_square_half_width_bits": half_width,
        "full_space_gap_lower_bound": np.floor(gap * 1e5) / 1e5,
        "max_field_edit_upper_bound": np.ceil(field * 1e5) / 1e5,
        "grid_cells": interior["cells_verified"],
        "refined_edge_cells": 0 if edge is None else edge["grid_edge_cells"],
        "max_grid_site_z_drift": interior["max_grid_local_drift"],
        "face_margins_bits": margins,
    }
    if not (half_width >= 0.05 and gap > GAP_FLOOR
            and field < FIELD_CAP):
        raise RuntimeError(f"Seed {interior['seed']} failed the requested coverage check")
    return result


def compensated_paths():
    endpoints = {str(seed): [opposing_endpoint(seed)] for seed in (3801, 3802)}
    bonds = xxz_bonds()
    site_terms = np.array([tensor_operator(Z, i) for i in range(8)])
    rows, audit = [], {}
    for seed in (3801, 3802):
        p0 = polarized_reference(seed)
        target, jacobian, _, _, _ = polarized_state(p0)
        opposing = next(r for r in endpoints[str(seed)] if r["target_bits"] == [.05, -.05])
        bond_edits = np.array(opposing["bond_edits"])
        derivative = compensated_response(jacobian)
        fd = np.column_stack([
            (polarized_state(compensated_state(p0, 1e-5*unit, target))[0][-2:]
             - polarized_state(compensated_state(p0, -1e-5*unit, target))[0][-2:]) / 2e-5
            for unit in np.eye(2)
        ])
        errors = {"max_compensated_derivative_error": np.max(abs(fd-derivative)),
                  "max_entropy_crosscheck_error_bits": 0.,
                  "max_gap_crosscheck_error": 0.,
                  "max_local_density_matrix_drift": 0.,
                  "max_ground_eigenpair_residual": 0.}
        for t in np.linspace(0, 1, 33):
            p = compensated_state(p0, t*bond_edits, target)
            result = polarized_state(p)
            for compensated in (True, False):
                state = p.copy()
                if not compensated:
                    state[:7] = p0[:7]
                field = HELMERT @ state[:7]
                h = np.einsum("i,ijk->jk", state[7:], bonds) + np.einsum("i,ijk->jk", field, site_terms)
                energies, basis = eigh(h, subset_by_index=[0, 1], check_finite=False)
                ground = basis[:, 0]
                single = np.array([reduced_state(ground, (i,)) for i in range(8)])
                z = single[:, 0, 0]-single[:, 1, 1]
                if t == 0:
                    reference = single
                rho = [reduced_state(ground, block) for block in BLOCKS]
                entropy = -np.log2([np.vdot(m, m).real for m in rho])
                gap = energies[1]-energies[0]
                matrix_drift = np.max(np.abs(single-reference))
                field_edits = field-HELMERT@p0[:7]
                rows.append({
                    "seed": seed, "fraction": t, "compensated": compensated,
                    "bond_a": t*bond_edits[0], "bond_b": t*bond_edits[1],
                    "delta_a": (entropy-target[-2:])[0], "delta_b": (entropy-target[-2:])[1],
                    "max_site_matrix_entry_drift": matrix_drift, "full_gap": gap,
                    "max_field_edit": np.max(abs(field_edits)),
                    **{f"z_{i}": z[i] for i in range(8)},
                    **{f"field_edit_{i}": field_edits[i] for i in range(8)},
                })
                if compensated:
                    levels = full_space_levels(p)
                    comparisons = {
                        "max_entropy_crosscheck_error_bits": np.max(abs(result[0][-2:]-entropy)),
                        "max_gap_crosscheck_error": np.max(abs(energies-levels)),
                        "max_local_density_matrix_drift": matrix_drift,
                        "max_ground_eigenpair_residual": np.linalg.norm(h@ground-energies[0]*ground),
                    }
                    for key, value in comparisons.items():
                        errors[key] = max(errors[key], value)
        if errors["max_compensated_derivative_error"] > 1e-8 or any(
            value > 1e-10 for key, value in errors.items()
            if key != "max_compensated_derivative_error"
        ):
            raise RuntimeError(f"Independent tensor basis check failed for seed {seed}: {errors}")
        audit[str(seed)] = errors

    return pd.DataFrame(rows), audit


def entropy_ceiling(bloch,n,block):
    """Bound using the minimum diagonal purity compatible with two marginals."""
    mags=np.linalg.norm(np.asarray(bloch).reshape(-1,n),axis=0)
    a,b=(1-mags[list(block)])/2
    t=np.clip((a+b)/2-.25,max(0,a+b-1),min(a,b))
    probs=np.array([1-a-b+t,a-t,b-t,t])
    sites=np.setdiff1d(np.arange(n),block)
    q=1-np.sum((1-mags[sites])/2)
    best=max(np.dot(probs,probs),max(0,q)**2)
    return -np.log2(best)


class LargeXXZ(Chain):
    """Iterative ground state and analytic tangent equation in a fixed sector."""
    def __init__(self,n):
        self.n,self.family=n,'XXZ'
        self.states=np.arange(2**n)
        bits=(self.states[:,None]>>np.arange(n))&1
        self.sectors=[np.flatnonzero(bits.sum(axis=1)==k) for k in range(n+1)]
        self.basis=self.sectors[n//2]
        self.z=[pauli_product(n,zmask=1<<i) for i in range(n)]
        zz=[pauli_product(n,zmask=3<<i) for i in range(n-1)]
        self.fields=[z-self.z[-1] for z in self.z[:-1]]
        bonds=[pauli_product(n,xmask=3<<j)@(eye(2**n)-v)+.8*v for j,v in enumerate(zz)]
        self.ops=self.fields+bonds
        self.nb=self.nf=n-1
        self.op_reduced=[a[self.basis][:,self.basis] for a in self.ops]
        self.audit_diag=1-2*bits[self.basis].T
        self.dim=len(self.basis)
        self._build_reduction_masks()
        self.start_vector=np.random.default_rng(n).normal(size=self.dim)

    def analyze(self,par,blocks):
        h=sum((x*a for x,a in zip(par,self.op_reduced)),csr_matrix((self.dim,self.dim)))
        w,v=eigsh(h,k=2,which='SA',tol=5e-13,v0=self.start_vector)
        psi=v[:,0]
        pencil=h-w[0]*eye(self.dim)
        col=csr_matrix(psi[:,None])
        factor=splu(bmat([[pencil,col],[col.T,None]],format='csc'))
        applied=np.array([a@psi for a in self.op_reduced]).T
        dv=factor.solve(np.vstack((-applied,np.zeros(len(par)))))[:-1]
        z=self.audit_diag@psi**2
        dc=2*(self.audit_diag[:-1]*psi)@dv
        s=[]; ds=[]
        for block in blocks:
            row,col=self.maps[tuple(sorted(block))]
            array=np.zeros((4,2**(self.n-2)))
            array[row,col]=psi
            rho=array@array.T
            p=np.sum(rho*rho)
            s.append(-np.log2(p))
            ds.append(-4*(rho@array)[row,col]@dv/(p*np.log(2)))
        return np.r_[z[:-1],s],np.vstack((dc,ds)),psi,z,w

    def full_gap(self,p,w):
        from scipy.optimize import linprog
        h=sum((a*b for a,b in zip(p,self.ops)),csr_matrix(self.ops[0].shape))
        diffs=[]
        for k,ids in enumerate(self.sectors):
            if k==self.n//2: continue
            x=h[ids][:,ids]
            e=eigsh(x,k=1,which='SA',tol=5e-12,v0=np.random.default_rng(k).normal(size=len(ids)),return_eigenvectors=False)[0] if len(ids)>2 else x.diagonal().min()
            diffs.append((self.n-2*k,e-w[0]))
        slopes,offsets=np.array(diffs).T
        fit=linprog([0,-1],A_ub=np.c_[-slopes,np.ones(len(slopes))],b_ub=offsets,
                    bounds=[(None,None),(None,w[1]-w[0])],method='highs')
        if not fit.success: raise RuntimeError(fit.message)
        return fit.x[1],fit.x[0]


def physical_fields(model,p,h):
    f=p[:model.nf]
    return np.r_[f,-f.sum()]+h if model.family=='XXZ' else f


def gap_for_bounded_fields(model,p,levels,initial,db):
    if model.family!='XXZ':
        return model.full_gap(p,levels)
    f=physical_fields(model,p,0)
    low=np.max(initial-db-f)
    high=np.min(initial+db-f)
    if low>high+1e-11:
        return -np.inf,(low+high)/2
    high=max(low,high)
    from scipy.optimize import linprog
    h=sum((v*a for v,a in zip(p,model.ops)),csr_matrix(model.ops[0].shape))
    diffs=[]
    for k,b in enumerate(model.sectors):
        if k==model.n//2: continue
        matrix=h[b][:,b]
        if len(b)>180:
            e=eigsh(matrix,k=1,which='SA',tol=5e-12,return_eigenvectors=False,
                    v0=np.random.default_rng(k).normal(size=len(b)))[0]
        else:
            e=eigh(matrix.toarray(),eigvals_only=True,subset_by_index=[0,0])[0]
        diffs.append((model.n-2*k,e-levels[0]))
    slopes,offsets=np.array(diffs).T
    result=linprog([0,-1],A_ub=np.c_[-slopes,np.ones(len(slopes))],b_ub=offsets,
                   bounds=[(low,high),(None,levels[1]-levels[0])],method='highs')
    if not result.success: raise RuntimeError(result.message)
    # Among solutions that maximize the gap, avoid unnecessary uniform field changes.
    gap=result.x[1]
    lower=max(low,max((gap-o)/s for s,o in zip(slopes,offsets) if s>0))
    upper=min(high,min((gap-o)/s for s,o in zip(slopes,offsets) if s<0))
    preferred=np.mean(initial-f)
    best=np.clip(preferred,min(lower,upper),max(lower,upper))
    return gap,best


def follow(model,start,blocks,goal,bon=.4,fld=.5,count=5,method='dogbox',perturb=None):
    value,_,_,audit,levels=model.analyze(start,blocks)
    base_gap,offset=model.full_gap(start,levels)
    initial=physical_fields(model,start,offset)
    if base_gap<.02:
        return "initial gap",0,[],base_gap
    if np.any(value[-2:]+goal<-1e-12):
        return "negative target",0,[],base_gap
    ceiling=np.array([entropy_ceiling(audit,model.n,block) for block in blocks])
    if np.any(value[-2:]+goal>ceiling+1e-12):
        return "fixed marginals",0,[],base_gap
    widths=np.r_[np.full(model.nf,fld),np.full(model.nb,bon)]
    lo=start-widths+1e-13
    hi=start+widths-1e-13
    history=[(start.copy(),offset,base_gap,0.,0.,0.,0.)]
    p=start.copy()
    last=None
    cached=None
    def cache(x):
        nonlocal last,cached
        if last is None or not np.array_equal(last,x):
            cached=model.analyze(x,blocks)
            last=x.copy()
        return cached
    scale=np.r_[np.ones(len(value)-2), np.clip(0.05/np.maximum(1e-5,np.abs(goal)),1,100)]
    # A fixed reference scale for constraints preserves the exact goal when attainable.
    scale[-2:]=max(scale[-2:]) if np.all(goal!=0) else min(scale[-2:]) if np.max(abs(goal))>=.01 else min(100,.05/max(1e-5,np.max(abs(goal))))
    for k in range(1,count+1):
        target=value.copy()
        target[-2:]+=goal*k/count
        curr,jac,*_=cache(p)
        r,d=model.reduced(jac)
        trial=p+d@np.linalg.pinv(r,rcond=1e-11)@(target[-2:]-curr[-2:])
        if perturb is not None and k==1:
            trial+=perturb*widths
        trial=np.clip(trial,lo+1e-12,hi-1e-12)
        ans=least_squares(lambda x: (cache(x)[0]-target)*scale,trial,
                          jac=lambda x: cache(x)[1]*scale[:,None],bounds=(lo,hi),
                          gtol=5e-13,xtol=1e-12,ftol=1e-12,max_nfev=65,method=method)
        p=ans.x
        realized,_,_,obs,levels=cache(p)
        er=np.max(abs(realized[-2:]-target[-2:]))
        drift=np.max(abs(obs-audit))
        gap,h=gap_for_bounded_fields(model,p,levels,initial,fld)
        physical_delta=np.max(abs(physical_fields(model,p,h)-initial))
        db=np.max(abs(p[model.nf:]-start[model.nf:]))
        history.append((p.copy(),h,gap,er,drift,db,physical_delta))
        relative_tol=min(1e-7,1e-3*np.max(np.abs(goal)))
        if drift>1e-7 or er>relative_tol:
            return "target residual",k-1,history,base_gap
        if physical_delta>fld+1e-9:
            return "field bound",k-1,history,base_gap
        if gap<.02:
            return "gap",k-1,history,base_gap
    return "pass",count,history,base_gap


def goals(tag,value,audit,n,blocks,corner=False):
    s=value[-2:]
    if tag=="absolute":
        down=up=np.full(2,.05)
    else:
        bounds=np.array([entropy_ceiling(audit,n,block) for block in blocks])
        down=np.minimum(.05,.1*s)
        up=np.minimum(.05,.1*np.maximum(0,bounds-s))
    for k,sign in product(range(2),(1,-1)):
        p=np.zeros(2)
        p[k]=up[k] if sign>0 else -down[k]
        yield f'{"AB"[k]}{"+" if sign>0 else "-"}',p
    if corner:
        for a,b in product((0,1),repeat=2):
            yield ('+' if a else '-')+('+' if b else '-'),np.array([up[0] if a else -down[0],up[1] if b else -down[1]])


def ensembles():
    return [('XXZ',6,1103,6),('XXZ',8,1123,6),
            ('Ising',6,1151,6),('Ising',7,1171,6),
            ('Ising-symmetric',6,1601,6),('Ising-symmetric',7,1631,6),
            ('XY',6,1213,6),('XY',8,1231,6),
            ('XY-separated',6,1423,6),('XY-separated',8,1451,6)]


def configurations(n):
    near=((0,1),(3,4))
    interleaved=((0,n-2),(1,n-1))
    mixed=((n//2-1,n//2),(0,n-1))
    return (('adjacent blocks',near),('interleaved pairs',interleaved),('mixed geometry',mixed))


def dense_h(model,p,h):
    dense=np.tensordot(p,np.array([a.toarray() for a in model.ops]),axes=1)
    if model.family=='XXZ':
        dense+=h*sum(op.toarray() for op in model.z)
    return dense


def certify_path(model, points, blocks, goal):
    p0,_,*_=points[0]
    vals,_,_,ref,_=model.analyze(p0,blocks)
    gap=np.inf
    error=local=0.
    count=0
    for k,(aa,bb) in enumerate(zip(points[:-1],points[1:])):
        a,h0,g0,*_=aa
        b,h1,g1,*_=bb
        norm=np.max(abs(eigvalsh(dense_h(model,b-a,h1-h0),check_finite=False)))
        subdivisions=max(8,int(np.ceil(norm/(max(1e-7,min(g0,g1)-.02)*.50))))
        for i in range(subdivisions+1):
            t=i/subdivisions
            p=(1-t)*a+t*b
            h=(1-t)*h0+t*h1
            level=eigvalsh(dense_h(model,p,h),subset_by_index=[0,1],check_finite=False)
            # Distance to the closest sampled Hamiltonian is at most norm/(2*subdivisions).
            gap=min(gap,level[1]-level[0]-norm/subdivisions-1e-9)
            values,_,_,obs,_=model.analyze(p,blocks)
            local=max(local,np.max(abs(obs-ref)))
            target=vals[-2:]+goal*(k+t)/(len(points)-1)
            error=max(error,np.max(abs(values[-2:]-target)))
            count+=1
    return gap,local,error,count


TILES = 8
ENTROPY_ALLOWANCE = 1e-9


def candidate_pairs(seed):
    initial = polarized_reference(seed)
    reference, jacobian, _, polarizations, _ = polarized_state(initial)
    field_jacobian = jacobian[:7, :7]
    candidates = []
    for pair in combinations(range(7), 2):

        response = compensated_response(jacobian, pair)
        smallest = float(np.linalg.svd(response, compute_uv=False)[-1])
        if smallest <= 1e-10:
            continue
        edits = np.linalg.solve(response, TARGET)
        field_response = -np.linalg.solve(field_jacobian, jacobian[:7, 7 + np.array(pair)])
        field = float(np.max(np.abs(HELMERT @ (field_response @ edits))))
        if max(abs(edits)) > BOND_CAP or abs(edits.mean()) > MEAN_CAP or field > FIELD_CAP:
            continue
        candidates.append({
            'pair': pair, 'predicted_difference': float(abs(edits[0] - edits[1])),
            'sigma_min': smallest, 'predicted_field': field,
        })
    if not candidates:
        return None, None, float(min(abs(polarizations))), 0
    lowest = min(candidates, key=lambda row: (row['predicted_difference'], row['pair']))
    largest = min(candidates, key=lambda row: (-row['sigma_min'], row['pair']))
    return lowest, largest, float(min(abs(polarizations))), len(candidates)


def certificate_at(point, initial, reference, half_widths, pair):
    value, jacobian, _, _, sector_levels = polarized_state(point)
    low = full_space_levels(point)
    if abs(low[0] - sector_levels[0]) >= 1e-10:
        return value, jacobian, low, None
    residual = np.linalg.norm(value[:7] - reference[:7])
    try:
        proof = certify_cell(point, jacobian, low, half_widths, initial, residual, pair)
    except AssertionError:
        proof = None
    return value, jacobian, low, proof


def witness_at(seed, pair, diff_cap, half):
    """Existence follows from certified opposite faces in a local box."""

    initial = polarized_reference(seed)
    reference = polarized_state(initial)[0]

    def residual(edits):
        p = compensated_state(initial, edits, reference, pair=pair)
        values, jacobian, _, _, _ = polarized_state(p)
        return values[-2:] - reference[-2:] - TARGET, compensated_response(jacobian, pair)

    try:
        result = root(residual, np.zeros(2), jac=True, tol=1e-10)
        edits = result.x
        if np.max(abs(residual(edits)[0])) > 1e-10:
            return {'witness_status': 'solver_unresolved'}
        p = compensated_state(initial, edits, reference, pair=pair)
    except (ValueError, np.linalg.LinAlgError):
        return {'witness_status': 'solver_unresolved'}

    difference = float(abs(edits[0] - edits[1]))
    diagnostic = {'center_difference': difference, 'bond_edits': edits.tolist()}
    if (difference + 2 * half > diff_cap or np.max(np.abs(edits)) + half > BOND_CAP
            or abs(np.mean(edits)) + half > MEAN_CAP):
        return {'witness_status': 'outside_witness_budget', **diagnostic}

    val, jac, lev, proof = certificate_at(p, initial, reference, np.full(2, half), pair)
    if proof is None or proof[0] < GAP_FLOOR or proof[1] > FIELD_CAP:
        return {'witness_status': 'box_unresolved', **diagnostic}

    inverse = np.linalg.inv(compensated_response(jac, pair))
    margins = []
    for axis, sign in product(range(2), (-1, 1)):
        face_margin = np.inf
        # Each face is covered by 256 closed intervals in the free coordinate.
        tiles = 256
        tile_half_width = half / tiles
        widths = np.full(2, tile_half_width)
        widths[axis] = 0.
        for j in range(tiles):
            offset = np.zeros(2)
            offset[axis] = sign * half
            offset[1-axis] = -half + (2*j+1)*tile_half_width
            face = compensated_state(initial, edits + offset, reference, guess=p[:7], pair=pair)
            fval, _, _, fproof = certificate_at(face, initial, reference, widths, pair)
            if fproof is None or fproof[0] < GAP_FLOOR or fproof[1] > FIELD_CAP:
                return {'witness_status': 'face_unresolved', **diagnostic}
            center_margin = sign * (inverse @ (fval[-2:] - reference[-2:] - TARGET))[axis]
            allowance = np.sum(np.abs(inverse[axis])) * fproof[2]
            face_margin = min(face_margin, center_margin - allowance)
        margins.append(float(face_margin))
    return {
        'witness_status': 'certified' if min(margins) > 0 else 'sign_unresolved',
        **diagnostic, 'certified_upper': difference + 2*half, 'witness_half_width': half,
        'witness_gap_lower': float(proof[0]), 'witness_field_upper': float(proof[1]),
        'center_gap': float(lev[1]-lev[0]), 'face_margins': margins,
        'center_target_error': float(max(abs(val[-2:] - reference[-2:] - TARGET))),
        'center_local_error': float(max(abs(val[:7] - reference[:7]))),
    }


def witness(seed, pair, diff_cap):
    # The entropy allowance in a very small box can dominate the face signal.
    # If the signs cannot be certified, check a larger box with the same caps.
    for half in (0.00005, 0.0001, 0.0002):
        answer = witness_at(seed, pair, diff_cap, half)
        if answer['witness_status'] != 'sign_unresolved':
            return answer
    return answer


def exclusion_tile(task):
    """Cover a square tile with safe entropy bounds or mark it unresolved."""
    seed, pair, cap, row, col = task

    initial = polarized_reference(seed)
    reference = polarized_state(initial)[0]
    half = BOND_CAP / TILES
    start = -BOND_CAP + (1 + 2*np.array([row, col]))*half
    stack = [(start, np.full(2, half), 0, None)]
    tested = off_strip = final = unresolved = 0
    minimum = np.inf
    while stack:
        center, widths, depth, warm = stack.pop()
        # Work on a superset of the admissible domain by omitting the mean cap.
        if abs(center[0]-center[1]) - widths.sum() > cap + 1e-13:
            off_strip += 1
            continue
        tested += 1
        if tested > 18000:
            unresolved += len(stack) + 1
            break
        try:
            comp = compensated_state(initial, center, reference, guess=warm, pair=pair)
        except (ValueError, np.linalg.LinAlgError):
            comp = None
        if comp is not None:
            value, _, _, proof = certificate_at(comp, initial, reference, widths, pair)
            if proof is not None and proof[0] > GAP_FLOOR:
                margin = np.max(np.abs(value[-2:] - reference[-2:] - TARGET)) - proof[2]
                if margin > ENTROPY_ALLOWANCE:
                    final += 1
                    minimum = min(minimum, margin)
                    continue
        if depth >= 40:
            unresolved += 1
            continue
        axis = int(np.argmax(widths))
        smaller = widths.copy()
        smaller[axis] *= 0.5
        delta = np.eye(2)[axis] * smaller[axis]
        guess = warm if comp is None else comp[:7]
        stack.extend([(center-delta, smaller, depth+1, guess),
                      (center+delta, smaller, depth+1, guess)])
    return {'row':row, 'col':col, 'tested':tested, 'off_strip':off_strip,
            'certified_cells':final, 'unresolved':unresolved,
            'minimum_margin':float(minimum) if final else None}


def summarize_exclusion(rows):
    margins = [row['minimum_margin'] for row in rows if row['minimum_margin'] is not None]
    return {'is_excluded':len(rows) == TILES*TILES and all(row['unresolved'] == 0 for row in rows),
            'tested':sum(row['tested'] for row in rows),
            'certified_cells':sum(row['certified_cells'] for row in rows),
            'minimum_margin':min(margins) if margins else None}


BLUE, TEAL, CLAY, LILAC, GRAY = "#567d98", "#4b857b", "#995f53", "#827596", "#6f7480"
PALE = "#dfede9"
PALETTE = [BLUE, TEAL, CLAY, LILAC, GRAY]
ENT_A, ENT_B = r"$\Delta S_2(A)$ (bits)", r"$\Delta S_2(B)$ (bits)"
AX_FORMAT = {"XXZ":"XXZ", "Ising":"Tilted Ising", "Ising-symmetric":"Symmetric Ising",
             "XY":"Linked XY", "XY-separated":"Separate XY"}


def canvas():
    fig, axes = plt.subplots(2, 2, figsize=(8.35, 6.15), layout="constrained")
    for index, axis in enumerate(axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        axis.annotate("abcd"[index], (0, 1), xycoords="axes fraction",
                      xytext=(-11, 7), textcoords="offset points",
                      fontsize=12, fontweight="bold", annotation_clip=False)
    return fig, axes.ravel()


def annotate_bar(axis, bars, labels):
    for bar, label in zip(bars, labels):
        axis.annotate(label, (bar.get_x()+bar.get_width()/2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)


def target_marker(ax, target, reachable):
    ax.plot(*target, marker="o" if reachable else "x", color=CLAY, ms=7, ls="none",
            markerfacecolor="white", markeredgewidth=1.55, zorder=5,
            label="Reached target" if reachable else "Excluded target")


def entropy_ax(ax, extent=.18):
    ax.set(xlabel=ENT_A, ylabel=ENT_B, xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    step = .1 if extent > .16 else .05
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.axhline(0, color=".82", lw=.65, zorder=0)
    ax.axvline(0, color=".82", lw=.65, zorder=0)


def independent(ax, data, width, target, extent):
    entropy_ax(ax, extent)
    ax.add_patch(Rectangle((-width, -width), 2*width, 2*width, facecolor=PALE, edgecolor=TEAL,
                           lw=1.1, label=f"Certified square (±{width:g} bits)"))
    ax.scatter(data.delta_a, data.delta_b, color=BLUE, s=7, alpha=.62, lw=0, label="Computed points")
    target_marker(ax, target, True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)


def narrow(ax, data, upper, target, extent):
    entropy_ax(ax, extent)
    x = np.array(ax.get_xlim())
    ax.fill_between(x, x-upper, x+upper, color=PALE, label=f"Bound on difference ({upper:.5f} bits)")
    ax.plot(x, x-upper, x, x+upper, color=TEAL, lw=.85)
    ax.scatter(data.delta_a, data.delta_b, color=BLUE, s=8, alpha=.66, lw=0, label="Computed points")
    target_marker(ax, target, False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)


def fig1(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    position = np.arange(8)
    ax.plot(position, np.zeros(8), color=".68", lw=1.1)
    for left, color, label in [(1, TEAL, "A"), (5, BLUE, "B")]:
        ax.add_patch(Rectangle((left-.36, -.22), 1.72, .32, color=color, alpha=.15))
        ax.plot([left, left+1], [0, 0], lw=3, color=color)
        ax.text(left+.5, .15, label, color=color, ha="center", va="bottom", fontweight="bold")
    ax.scatter(position, np.zeros(8), color=".22", s=23, zorder=3)
    for k in position:
        ax.text(k, -.31, str(k), ha="center")
    ax.set(xlim=(-.55, 7.55), ylim=(-.52, .43), title="Open eight-spin XXZ chain")
    ax.set_axis_off()
    selects = data["compensated_path"].query("seed == 3802 and compensated == True")
    begin, end = selects.iloc[0], selects.iloc[-1]
    bx.plot(position, [begin[f"z_{i}"] for i in position], "o-", color=BLUE, label="Starting state")
    bx.scatter(position, [end[f"z_{i}"] for i in position], facecolors="none", edgecolors=CLAY,
               s=53, label="After compensation")
    bx.axhline(0, lw=.7, color=".8")
    bx.set(xlabel="Site", ylabel=r"$\langle Z_i\rangle$", xticks=position, title="One-spin states")
    bx.legend(loc="upper center", bbox_to_anchor=(.5, -.21), ncol=2)

    bound = audit["polarized"]["restricted"]
    width = audit["polarized"]["coverage"][1]["independent_bond_edit_half_width"]
    cx.add_patch(Rectangle((-width, -width), 2*width, 2*width, facecolor="#e9eef2", edgecolor=BLUE, lw=1))
    radius = bound["common_limit"]
    narrow_width = bound["differential_limit"]
    corners = np.array([[-radius, -narrow_width], [-radius, narrow_width],
                        [radius,narrow_width], [radius,-narrow_width]]) @ np.array([[1, 1], [1, -1]])
    cx.fill(*corners.T, color=TEAL, alpha=.9, label="Restricted edits")
    cx.plot([], [], color=BLUE, label="Independent edits")
    cx.set(xlabel=r"Bond edit $\delta J_A$", ylabel=r"Bond edit $\delta J_B$", title="Control domains",
           xlim=(-.174, .174), ylim=(-.174, .174), aspect="equal")
    cx.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
    entropy_ax(dx, .115)
    dx.set_title("Entropy targets")
    for a, b in [(.05, 0), (-.05, 0), (0,.05), (0,-.05)]:
        dx.plot(a,b, "o", color=BLUE, ms=5)
    dx.plot(0, 0, "+", color=".35", ms=7)
    dx.scatter([-.05,-.05,.05], [-.05,.05,.05], marker="s", color=TEAL, s=24)
    dx.plot(.05, -.05, "s", color=CLAY, ms=6)
    return fig


def fig2(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    s = audit["symmetric"]
    grid = data["symmetric_grid"]
    values = grid.bond_a.unique()[::4]
    sampled = grid[grid.bond_a.isin(values) & grid.bond_b.isin(values)]
    ax.set_title("Independent bond edits")
    bx.set_title("Restricted edit difference")
    independent(ax, sampled, s["square_half_width_bits"], s["target_bits"], .184)
    narrow(bx, data["symmetric_restricted"], s["restricted_difference_bound_bits"],
           s["target_bits"], .184)
    extent = s["bond_half_width"]
    heat = cx.imshow(grid.gap.to_numpy().reshape(s["grid"], -1), origin="lower",
                     extent=(-extent,extent,-extent,extent), cmap="cividis")
    fig.colorbar(heat, ax=cx, label="Sampled full-space gap", shrink=.8)
    cx.set(xlabel=r"Bond edit $\delta J_B$", ylabel=r"Bond edit $\delta J_A$",
           title="Full-space gap")
    margins = np.array(s["face_margins_bits"]).ravel()
    x = np.arange(4)
    dx.bar(x, margins, color=[TEAL, TEAL, BLUE, BLUE], width=.64)
    dx.axhline(.1, color=CLAY, ls="--", label="0.10-bit target half-width")
    dx.set(xticks=x, xticklabels=["A−","A+","B−","B+"], ylim=(0,.135),
           ylabel="Certified face margin (bits)", title="Opposite-face sign test")
    dx.legend(loc="upper right")
    return fig


def fig4(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    path = data["compensated_path"]
    for seed, color in [(3801, BLUE), (3802, TEAL)]:
        selected = path[path.seed.eq(seed)]
        corrected, bare = selected[selected.compensated], selected[~selected.compensated]
        t = corrected.fraction
        ax.plot(t, corrected.delta_a, color=color, label=f"{seed}, A")
        ax.plot(t, corrected.delta_b, color=color, ls="--", label=f"{seed}, B")
        bx.semilogy(t.iloc[1:], corrected.max_site_matrix_entry_drift.iloc[1:].clip(lower=1e-16),
                    color=color, label=f"{seed}, compensated")
        bx.semilogy(t.iloc[1:], bare.max_site_matrix_entry_drift.iloc[1:],
                    color=color, ls="--", label=f"{seed}, fields held fixed")
        cx.plot(t, corrected.max_field_edit, color=color, label=str(seed))
        dx.plot(t, corrected.full_gap, color=color, label=str(seed))
        bound = next(r["full_space_gap_lower_bound"] for r in audit["polarized"]["coverage"]
                     if r["seed"] == seed)
        dx.axhline(bound, color=color, ls="--", lw=.85)
    ax.set_ylabel("Entropy change (bits)")
    ax.set_title("Opposing entropy changes")
    bx.set_ylabel("Maximum one-spin matrix drift")
    bx.set_title("Effect of field compensation")
    cx.axhline(.30, color=CLAY, ls="--", label="Field edit limit")
    cx.set(ylim=(0,.33), ylabel="Maximum site-field edit", title="Compensating field edits")
    dx.set(ylabel="Full-space gap", title="Gap along the sampled bond line", ylim=(.65,1.3))
    for axis in (ax,bx,cx,dx):
        axis.set_xlabel("Fraction of bond edit")
    ax.legend(ncol=2, loc="upper left")
    bx.legend(ncol=2, loc="center right")
    cx.legend(loc="center left")
    dx.legend(loc="upper right")
    return fig


def groupbars(ax, frame, items, index, kind, labelmap):
    for offset, mode, color in [(-.17,"absolute",BLUE), (.17,"scaled",TEAL)]:
        chosen = frame[frame[kind].eq(mode)].set_index(index).reindex(items)
        values = chosen.cases_passing/chosen.cases_tested
        bars = ax.bar(np.arange(len(items))+offset, values, width=.32, color=color,
                      label="0.05-bit" if mode == "absolute" else "Scaled")
        annotate_bar(ax, bars, [f"{a}/{b}" for a,b in zip(chosen.cases_passing, chosen.cases_tested)])
    ax.set(xticks=np.arange(len(items)), xticklabels=[labelmap.get(x,x) for x in items],
           ylim=(0,1.17), ylabel="Cases passing all four axes")
    ax.set_yticks(np.linspace(0,1,5))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def fig5(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    budgets = data["budget_screen"].query("model == 'XXZ'")
    for (budget, field), group in budgets.groupby(["budget", "field_budget"], sort=False):
        color = {0.4: BLUE, 0.2: TEAL, 0.1: CLAY}[budget]
        ax.plot(group.n, group.cases_passing/group.cases_tested, "o-", color=color,
                label=f"{budget:.2f} / {field:.2f}")
    ax.set(xlabel="Spins", ylabel="Cases passing all four axes", ylim=(.55,1.11), xticks=[6,8,10,12])
    ax.set_yticks(np.linspace(.6,1,5))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(title="Bond / field cap", loc="lower left")
    corners = data["corner_screen"]
    bars = bx.bar(corners.n, corners.targets_passed/corners.targets_tested, width=.91, color=BLUE)
    annotate_bar(bx, bars, [f"{a}/{b}" for a,b in zip(corners.targets_passed,corners.targets_tested)])
    bx.set(xlabel="Spins", ylabel="Joint corner targets reached", ylim=(.0,1.13), xticks=[6,8,10,12])
    bx.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    groupbars(cx, data["geometry_screen"], GEOMETRIES, "geometry", "mode",
              {"adjacent blocks":"Adjacent", "interleaved pairs":"Interleaved", "mixed geometry":"Mixed"})
    cx.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, -.15))
    rows = data["screen"]
    rows = rows.query("model == 'XXZ' and mode == 'absolute' and budget == 0.4 and result == 'pass'")
    rng = np.random.default_rng(501)
    for n, data in rows.groupby("n"):
        dx.scatter(n+rng.uniform(-.16,.16,len(data)), data.gap_min, alpha=.44, s=12, color=BLUE, lw=0)
    dx.set(xlabel="Spins", ylabel="Minimum checkpoint gap per target", xticks=[6,8,10,12])
    ax.set_title("Axis targets across budgets")
    bx.set_title("Joint targets at 0.40 / 0.50 caps")
    cx.set_title("Block geometry")
    dx.set_title("Gaps on successful targets")
    return fig


def fig6(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    groupbars(ax, data["model_screen"], FAMILIES, "model", "mode",
              {"XXZ":"XXZ","Ising":"Tilted\nIsing","Ising-symmetric":"Symmetric\nIsing",
               "XY":"Linked\nXY","XY-separated":"Separate\nXY"})
    ax.legend(ncol=2, loc="upper right")
    tilted = data["tilted_absolute_targets"]
    for status, label, color in [("negative target","Below zero",BLUE),
                                  ("fixed marginals","Above ceiling",TEAL),
                                  ("target residual","Solver miss",CLAY)]:
        rows = tilted[tilted.result.eq(status)]
        bx.scatter(rows.ceiling_bits,rows.goal_entropy_bits,color=color,s=17,lw=0,alpha=.72,label=label)
    x = np.linspace(0, max(tilted.ceiling_bits.max(), tilted.goal_entropy_bits.max()), 100)
    bx.plot(x,x, ls="--", color=".4", lw=.8)
    bx.axhline(0, color=".6", lw=.8)
    bx.set(xlabel="Fixed-marginal ceiling (bits)", ylabel="Requested entropy (bits)")
    bx.legend(loc="lower right")
    cm = data["complementary_blocks"]
    cx.plot(cm.sample_bond_edit, cm.entropy_a, color=BLUE, lw=2, label=r"$S_2(A)$")
    cx.plot(cm.sample_bond_edit, cm.entropy_b, color=CLAY, ls="--", lw=1.25, label=r"$S_2(B)$")
    cx.set(xlabel="Sampled bond edit", ylabel=r"$S_2$ (bits)")
    cx.legend(loc="best")
    field = data["field_only"]
    dx.scatter(np.arange(len(field)), field.min_local_field_singular, color=TEAL, s=30)
    dx.set(yscale="log", xticks=np.arange(len(field)),
           xticklabels=[AX_FORMAT[f].replace(" ","\n",1) for f in field.family],
           ylabel="Smallest field-susceptibility singular value")
    ax.set_title("Family screen")
    bx.set_title("Tilted-Ising target exclusions")
    cx.set_title("Complementary blocks in a four-spin chain")
    dx.set_title("Fields alone at fixed local states")
    return fig


def fig_s1(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    panel = audit["polarized"]["coverage"]
    raw = [data["polarized_boxes"][seed] for seed in (3801, 3802)]
    xpos = np.arange(2)
    for x, row, full in zip(xpos, panel, raw):
        ax.plot([x,x], [row["full_space_gap_lower_bound"],full["min_sampled_full_space_gap"]],
                color=".65", lw=1.8, zorder=1)
    ax.scatter(xpos, [r["min_sampled_full_space_gap"] for r in raw], color=BLUE,
               marker="o", s=33, label="Minimum sampled", zorder=2)
    ax.scatter(xpos, [r["full_space_gap_lower_bound"] for r in panel], color=TEAL,
               marker="s", s=33, label="Continuous lower bound", zorder=2)
    ax.set(xticks=xpos, xticklabels=[3801,3802], xlim=(-.45,1.45), ylabel="Full-space gap")
    ax.legend(loc="center right")

    bars = bx.bar(xpos, [r["max_field_edit_upper_bound"] for r in panel],
                  color=[BLUE,TEAL], width=.57)
    annotate_bar(bx, bars, [f"{r['max_field_edit_upper_bound']:.5f}" for r in panel])
    bx.axhline(.3, color=CLAY, ls="--", label="Field edit limit")
    bx.set(xticks=xpos, xticklabels=[3801,3802], ylim=(0,.37), ylabel="Field edit upper bound")
    bx.legend(loc="upper left")

    widths = [min(np.ravel(raw[0]["face_margins"])), panel[0]["certified_square_half_width_bits"],
              panel[1]["certified_square_half_width_bits"]]
    bars = cx.bar(range(3), widths, color=[GRAY,BLUE,TEAL], width=.56)
    for index, (bar, width) in enumerate(zip(bars, widths)):
        y = width - .006 if index == 0 else width + .003
        cx.text(bar.get_x()+bar.get_width()/2, y, f"{width:.5f}",
                color="white" if index == 0 else ".15", ha="center", va="center", fontsize=7.5)
    cx.axhline(.05, color=CLAY, ls="--")
    cx.set(xticks=range(3), xticklabels=["3801\n151² faces","3801\nrefined edges","3802\n201² faces"],
           ylim=(0,.102), ylabel="Bound on square half-width (bits)")

    errors = audit["independent_tensor_checks"]
    keys = ["max_compensated_derivative_error","max_entropy_crosscheck_error_bits",
            "max_gap_crosscheck_error","max_local_density_matrix_drift"]
    x = np.arange(len(keys))
    for change, seed, color in [(-.15, "3801", BLUE), (.15, "3802", TEAL)]:
        dx.scatter(x+change, [errors[seed][key] for key in keys], color=color, label=seed, s=27)
    dx.set(yscale="log", xticks=x, xticklabels=["Jacobian","Entropy","Energy","Site state"],
           ylabel="Maximum absolute discrepancy", ylim=(1e-16,1e-8))
    dx.legend(loc="upper right", ncol=2)
    ax.set_title("All-sector gap checks")
    bx.set_title("Physical field edits")
    cx.set_title("Opposite-face refinement")
    dx.set_title("Independent numerical checks")
    return fig


def fig_s2(data, audit):
    fig, (ax, bx, cx, dx) = canvas()
    profiles = data["compensated_path"].query("seed == 3802 and compensated == True")
    for i in range(8):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(profiles.fraction, profiles[f"field_edit_{i}"], color=color,
                ls="-" if i < 5 else "--", label=str(i))
    ax.set(xlabel="Fraction of bond edit", ylabel="Longitudinal field edit")
    ax.legend(title="Site", ncol=4, loc="upper left")

    paths = data["path_summary"]
    fields = ["worst_sampled_drift","worst_sampled_target_error","smallest_gap_bound"]
    for (family, color) in [("XXZ", BLUE), ("Ising-symmetric", TEAL)]:
        linear = paths[paths.model.eq(family)]
        for axis, column in zip((bx,cx,dx), fields):
            axis.plot(linear["count"], linear[column], "o-", color=color, label=AX_FORMAT[family])
    for axis in (bx,cx,dx):
        axis.set(xscale="log", xticks=[5,25,125], xticklabels=[5,25,125],
                 xlabel="Continuation increments")
        axis.legend(loc="best")
    bx.set(yscale="log", ylabel="Maximum sampled local drift")
    cx.set(yscale="log", ylabel="Maximum sampled entropy error (bits)")
    dx.set(ylabel="Worst continuous gap lower bound", ylim=(.4,1.4))
    ax.set_title("Site-by-site compensation (3802)")
    bx.set_title("Between-checkpoint local drift")
    cx.set_title("Between-checkpoint target error")
    dx.set_title("Gap along interpolated paths")
    return fig


def fig_s3(data, audit):
    fig, (ax,bx,cx,dx) = canvas()
    cases = data["axis_case_status"]
    small = cases.query("model == 'XXZ' and n <= 8 and mode == 'absolute'")
    for axis, selection, title in [(ax, small, "XXZ coordinate-target cases"),
                                  (bx, cases.query("model == 'Ising-symmetric' and mode == 'absolute'"),
                                   "Symmetric Ising coordinate-target cases")]:
        table = selection.pivot(index=["n","seed","pair"], columns="budget", values="pass")
        table = table.sort_index().sort_index(axis=1, ascending=False)
        from matplotlib.colors import ListedColormap
        axis.imshow(table.T.astype(float), aspect="auto", interpolation="nearest",
                    cmap=ListedColormap(["#e7d7d0","#6a998e"]), vmin=0, vmax=1)
        axis.set(yticks=np.arange(len(table.columns)), yticklabels=[f"{r:.2f}" for r in table.columns],
                 xlabel="Case index (size, seed and block pair)", ylabel="Bond edit cap", title=title)
        if axis is bx:
            axis.legend(handles=[Rectangle((0,0),1,1,color="#6a998e",label="Pass"),
                                 Rectangle((0,0),1,1,color="#e7d7d0",label="Other outcome")],
                        loc="upper center", bbox_to_anchor=(.5,-.2), ncol=2)
    geometry = data["geometry"].query("mode == 'absolute' and direction == 'A+'")
    for i, (label,group) in enumerate(geometry.groupby("geometry",sort=False)):
        cx.scatter(i+np.linspace(-.15,.15,len(group)), group.smin, color=PALETTE[i], s=19, alpha=.8)
    cx.set(yscale="log", xticks=[0,1,2], xticklabels=["Adjacent","Interleaved","Mixed"],
           ylabel="Smaller compensated response singular value", title="Geometry and local response")
    restricted = audit["polarized"]["restricted"]
    control_names = ["Zero field","Polarized\n3802"]
    values = [audit["symmetric"]["restricted_normalized_singular_values"], restricted["normalized_singular_values"]]
    for i, color in [(0, BLUE), (1, TEAL)]:
        dx.bar(np.arange(2)+(i-.5)*.32, [v[i] for v in values], width=.30, color=color,
               label="Larger" if i == 0 else "Smaller")
    dx.set(yscale="log", xticks=[0,1], xticklabels=control_names,
           ylabel="Budget-normalized response singular value", title="Rank two under restricted edits")
    dx.legend(loc="best")
    return fig


PLACEMENT_BLUE = "#284e73"
PLACEMENT_TEAL = "#097f75"
PLACEMENT_CLAY = "#a04931"
PLACEMENT_GRAY = "#777a7b"


def rounded_up(value):
    return ceil(value * 1e6) / 1e6 if value is not None else None


def outcome(row, key):
    if row[key]['witness_status'] == 'certified':
        return 'reachable'
    if row[key].get('pair') == [1, 5] and row['baseline_status'] == 'excluded':
        return 'excluded'
    return 'inconclusive'


def panel_letter(ax, label):
    ax.text(0., 1.035, label, transform=ax.transAxes, fontweight='bold',
            fontsize=12, ha='left', va='bottom')


def entropy_panel(ax, samples, seed, square=None, exclusion=None):
    domain = 'independent' if square is not None else 'restricted'
    data = samples[(samples['seed'] == seed) & (samples['domain'] == domain)]
    if square is not None:
        ax.add_patch(Rectangle((-square,-square), 2*square, 2*square,
                               facecolor='#dcece8', edgecolor=PLACEMENT_TEAL, lw=1.2,
                               label=f'Certified square (±{square:.5f} bits)'))
        mark = ax.scatter(*TARGET, s=50, marker='o', edgecolors=PLACEMENT_CLAY,
                          facecolors='white', zorder=4, label='Reached target')
    else:
        x = np.linspace(-.142,.142,200)
        ax.fill_between(x,x-exclusion,x+exclusion,color='#dcece8',
                        label='Certified exclusion strip')
        mark = ax.scatter(*TARGET, s=50, marker='x', color=PLACEMENT_CLAY, zorder=4, label='Excluded target')
    pts = ax.scatter(data['delta_a'], data['delta_b'], s=3.5, color=PLACEMENT_BLUE, alpha=.52,
                     linewidths=0, zorder=2, label='Sampled settings')
    h, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[h[0], pts, mark], frameon=False, loc='upper left',
              fontsize=7.3, handletextpad=.45, borderaxespad=.35)
    ax.set(xlim=(-.142,.142),ylim=(-.142,.142), aspect='equal',
           xlabel=r'$\Delta S_2(A)$ (bits)',ylabel=r'$\Delta S_2(B)$ (bits)')
    ax.set_xticks([-.1,-.05,0,.05,.1])
    ax.set_yticks([-.1,-.05,0,.05,.1])


def pilot_panel(ax, rows):
    for y, row in enumerate(rows):
        lower = row['baseline_cap_checked']
        alt = row['differential']
        if alt['witness_status'] == 'certified':
            ax.scatter(alt['certified_upper'], y, marker='o', s=38, color=PLACEMENT_TEAL, zorder=4)
        if row['baseline_status'] == 'excluded':
            if 'baseline_upper' in row:
                upper = row['baseline_upper']['certified_upper']
                ax.plot([lower, upper], [y, y], color=PLACEMENT_CLAY, lw=2, zorder=5)
                # Two nearly coincident boundaries appear as a thin vertical mark.
                ax.scatter([lower, upper], [y, y], marker='|', s=100, color=PLACEMENT_CLAY, zorder=5)
                ax.annotate(f'({lower:.3f}, {rounded_up(upper):.6f}]', (lower, y),
                            xytext=(-5, -12), textcoords='offset points', ha='right',
                            va='center', color=PLACEMENT_CLAY, fontsize=7.5)
            else:
                ax.annotate('', (lower + .012, y), (lower, y),
                            arrowprops={'arrowstyle':'->','color':PLACEMENT_CLAY,'lw':1.45,
                                        'shrinkA':0,'shrinkB':0})
    ax.axvline(DIFF_CAP, lw=1, color=PLACEMENT_GRAY, ls='--', zorder=1)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[str(row['seed']) for row in rows],
           ylim=(len(rows)-.55, -.55), xlim=(0, .153),
           xlabel=r'Bond-edit difference cap $|\delta J_j-\delta J_k|$', ylabel='Pilot seed')
    ax.set_xticks([0, .025, .05, .075, .10, .125, .15])
    ax.grid(axis='x', lw=.55, color='.9')
    ax.legend(handles=[
        Line2D([],[],color=PLACEMENT_CLAY,marker='>',ls='none',label='Original pair: lower bound or bracket'),
        Line2D([],[],color=PLACEMENT_TEAL,marker='o',ls='none',label='Replacement pair: sufficient cap'),
    ], frameon=False, loc='upper right', fontsize=7.5)


def fig3(data, audit):
    pilot_rows = data["pilots"]
    samples = data["polarized_samples"]
    fig, axes = plt.subplots(2,2,figsize=(9.25,7.4), layout='constrained')
    entropy_panel(axes[0,0],samples,3802,audit["polarized"]["coverage"][1]["certified_square_half_width_bits"])
    entropy_panel(axes[0,1], samples, 3802, exclusion=audit["polarized"]["restricted"]["entropy_difference_offset_upper_bound"])
    entropy_panel(axes[1,0],samples,3801,audit["polarized"]["coverage"][0]["certified_square_half_width_bits"])
    pilot_panel(axes[1,1],pilot_rows)
    for ax,label in zip(axes.ravel(),'abcd'):
        panel_letter(ax,label)
    return fig


def fig_s4(data, audit):
    rows = data["held_out"]
    states = [[row['baseline_status'], outcome(row,'differential'), outcome(row,'sigma')]
              for row in rows]
    palette = {'reachable':PLACEMENT_TEAL,'excluded':PLACEMENT_CLAY,'unresolved':PLACEMENT_GRAY,'inconclusive':PLACEMENT_GRAY}
    markers = {'reachable':'o','excluded':'x','unresolved':'s','inconclusive':'s'}
    fig, ax = plt.subplots(figsize=(6.0,5.25),layout='constrained')
    for y, values in enumerate(states):
        for x, state in enumerate(values):
            ax.scatter(x,y,color=palette[state],marker=markers[state],s=55)
    ax.set(xlim=(-.5,2.5),ylim=(len(rows)-.45,-.55),xticks=[0,1,2],
           xticklabels=['Original pair','Differential rule','Singular-value rule'],
           yticks=np.arange(len(rows)),yticklabels=[str(row['seed']) for row in rows],
           ylabel='Held-out seed')
    ax.grid(axis='y', color='.92', lw=.55)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([],[],color=palette[k],ls='none',marker=markers[k],label=t)
                       for k,t in [('reachable','Certified reachable'),('excluded','Certified excluded'),
                                   ('inconclusive','Inconclusive')]],
              frameon=False, loc='lower center', bbox_to_anchor=(.5,1.014),ncol=3,fontsize=8)
    return fig


def opposing_endpoint(seed):
    initial = polarized_reference(seed)
    reference, *_ = polarized_state(initial)

    def residual(edits):
        state = compensated_state(initial, edits, reference)
        values, jacobian, *_ = polarized_state(state)
        return values[-2:] - reference[-2:] - TARGET, compensated_response(jacobian)

    result = root(residual, np.zeros(2), jac=True, tol=1e-11)
    state = compensated_state(initial, result.x, reference)
    levels = full_space_levels(state)
    sector_levels = polarized_state(state)[-1]
    entropy_error, _ = residual(result.x)
    if (np.max(np.abs(entropy_error)) >= 1e-10 or abs(levels[0] - sector_levels[0]) >= 1e-10
            or levels[1] - levels[0] <= GAP_FLOOR
            or np.max(np.abs(result.x)) >= {3801:.12, 3802:BOND_CAP}[seed]
            or np.max(np.abs(HELMERT @ (state[:7] - initial[:7]))) >= FIELD_CAP):
        raise RuntimeError(f"Unverified opposing endpoint: {seed}")
    return {"target_bits": TARGET.tolist(), "bond_edits": result.x.tolist()}


def polarized_plot_samples(coverage, restriction):
    rows = []
    for run in coverage:
        seed = run["seed"]
        width = run["independent_bond_edit_half_width"]
        grid = np.asarray(list(product(np.linspace(-width, width, 21), repeat=2)))
        offsets = sample_offsets(polarized_reference(seed), grid)
        rows.extend((seed, "independent", *edits, *result) for edits, result in zip(grid, offsets))
    mean = restriction["common_limit"]
    half_difference = restriction["differential_limit"]
    grid = np.asarray([EDIT_TRANSFORM @ (x, y) for x in np.linspace(-mean, mean, 81)
                       for y in (-half_difference, 0., half_difference)])
    offsets = sample_offsets(polarized_reference(3802), grid)
    rows.extend((3802, "restricted", *edits, *result) for edits, result in zip(grid, offsets))
    return pd.DataFrame(rows, columns=["seed", "domain", "bond_a", "bond_b", "delta_a", "delta_b"])


def screen_case(family, n, seed):
    chain = LargeXXZ(n) if n > 8 else Chain(family, n)
    start = chain.params(seed)
    ranges = [("absolute", .4, .5)]
    if n <= 8:
        ranges.append(("scaled", .4, .5))
    if family in ("XXZ", "Ising-symmetric"):
        ranges.append(("absolute", .2, .35))
        if n <= 8:
            ranges.append(("absolute", .1, .2))
    rows = []
    for pair, blocks in enumerate(block_configs(n)):
        values, _, _, observables, _ = chain.analyze(start, blocks)
        ceiling = [entropy_ceiling(observables, n, block) for block in blocks]
        for mode, bond, field in ranges:
            corners = family == "XXZ" and bond == .4 and mode == "absolute"
            for direction, target in goals(mode, values, observables, n, blocks, corner=corners):
                status, _, path, gap = follow(chain, start, blocks, target, bon=bond, fld=field)
                rows.append((family, n, seed, pair, mode, bond, field, direction, status,
                             min((step[2] for step in path), default=gap),
                             *values[-2:], *target, *ceiling))
    return rows


def geometry_case(n, seed):
    chain = Chain("XXZ", n)
    start = chain.params(seed)
    rows = []
    for geometry, blocks in configurations(n):
        values, jacobian, _, observables, _ = chain.analyze(start, blocks)
        response, _ = chain.reduced(jacobian)
        smallest = np.linalg.svd(response, compute_uv=False)[-1]
        for mode in ("absolute", "scaled"):
            for direction, target in goals(mode, values, observables, n, blocks):
                status, *_ = follow(chain, start, blocks, target)
                rows.append((n, seed, geometry, mode, direction, smallest, status))
    return rows


def path_case(family, n, seed, pair):
    chain = Chain(family, n)
    start = chain.params(seed)
    blocks = block_configs(n)[pair]
    values, _, _, observables, _ = chain.analyze(start, blocks)
    rows = []
    for direction, target in goals("absolute", values, observables, n, blocks, corner=True):
        for count in (5, 25, 125):
            status, _, path, _ = follow(chain, start, blocks, target, count=count)
            if status == "pass":
                gap, local, error, _ = certify_path(chain, path, blocks, target)
                rows.append((family, seed, direction, count, gap, local, error))
    return rows


def local_field_response():
    rows = []
    for family, n, seed in (("XXZ", 8, 1128), ("Ising-symmetric", 7, 1631),
                            ("Ising", 6, 1151), ("XY", 6, 1213)):
        chain = Chain(family, n)
        _, jac, *_ = chain.analyze(chain.params(seed), block_configs(n)[0])
        rows.append((family, np.linalg.svd(jac[:-2, :chain.nf], compute_uv=False)[-1]))
    return pd.DataFrame(rows, columns=["family", "min_local_field_singular"])


def complementary_entropies():
    chain = Chain("XXZ", 4)
    start = chain.params(1411)
    rows = []
    for edit in np.linspace(-.15, .15, 21):
        edited = start.copy()
        edited[-3] += edit
        values, *_ = chain.analyze(edited, ((0, 1), (2, 3)))
        rows.append((edit, *values[-2:]))
    data = pd.DataFrame(rows, columns=["sample_bond_edit", "entropy_a", "entropy_b"])
    np.testing.assert_allclose(data.entropy_a, data.entropy_b, atol=1e-12, rtol=0)
    return data


def axis_status(data, keys):
    rows = data[data.direction.isin(AXES)]
    grouped = rows.groupby(keys, sort=False)
    if not grouped.direction.agg(lambda values: len(values) == len(AXES) and set(values) == AXES).all():
        raise RuntimeError("Incomplete axis-target data")
    return grouped.result.agg(lambda values: (values == "pass").all()).rename("pass").reset_index()


def group_counts(rows, keys):
    return (rows.groupby(keys)["pass"].agg(["sum", "size"]).reset_index()
            .rename(columns={"sum": "cases_passing", "size": "cases_tested"}))


def continuation_figure_data(screen_jobs, geometry_jobs, path_jobs):
    screen = pd.DataFrame([row for job in screen_jobs for row in job.result()],
                          columns=["model", "n", "seed", "pair", "mode", "budget", "field_budget",
                                   "direction", "result", "gap_min", "sA", "sB", "target_A",
                                   "target_B", "upper_A", "upper_B"])
    geometry = pd.DataFrame([row for job in geometry_jobs for row in job.result()],
                            columns=["n", "seed", "geometry", "mode", "direction", "smin", "result"])
    paths = pd.DataFrame([row for job in path_jobs for row in job.result()],
                         columns=["model", "seed", "direction", "count", "gap_bound",
                                  "sampled_local_drift", "sampled_target_error"])
    passing = axis_status(screen, ["model", "n", "seed", "pair", "mode", "budget", "field_budget"])
    budgets = passing[passing["mode"].eq("absolute") & passing.model.isin(["XXZ", "Ising-symmetric"])]
    corners = screen[screen.direction.isin({"--", "-+", "+-", "++"})]
    corner_counts = corners.groupby("n").result.agg(
        targets_passed=lambda values: (values == "pass").sum(), targets_tested="size").reset_index()
    tilted = screen[(screen.model == "Ising") & (screen["mode"] == "absolute")].copy()
    target_a = tilted.direction.str.startswith("A")
    tilted["goal_entropy_bits"] = np.where(target_a, tilted.sA + tilted.target_A,
                                           tilted.sB + tilted.target_B)
    tilted["ceiling_bits"] = np.where(target_a, tilted.upper_A, tilted.upper_B)
    path_keys = ["model", "seed", "direction"]
    comparable = paths.groupby(path_keys)["count"].transform("nunique") == 3
    path_summary = (paths[comparable].groupby(["model", "count"])
                    .agg(worst_sampled_drift=("sampled_local_drift", "max"),
                         worst_sampled_target_error=("sampled_target_error", "max"),
                         smallest_gap_bound=("gap_bound", "min")).reset_index())
    return {
        "screen": screen,
        "axis_case_status": passing,
        "geometry": geometry,
        "budget_screen": group_counts(budgets, ["model", "n", "budget", "field_budget"]),
        "corner_screen": corner_counts,
        "geometry_screen": group_counts(axis_status(geometry, ["geometry", "mode", "n", "seed"]),
                                        ["geometry", "mode"]),
        "model_screen": group_counts(passing[passing.budget.eq(.4) & passing.n.le(8)], ["model", "mode"]),
        "tilted_absolute_targets": tilted,
        "path_summary": path_summary,
        "field_only": local_field_response(),
        "complementary_blocks": complementary_entropies(),
    }


def start_exclusion(pool, seed, cap):
    return [pool.submit(exclusion_tile, (seed, BASELINE, cap, i, j))
            for i, j in product(range(TILES), repeat=2)]


def selected_witnesses(seed):
    if seed in PILOT_PAIRS:
        candidates = [{"pair": PILOT_PAIRS[seed]}, None]
    else:
        differential, singular, _, _ = candidate_pairs(seed)
        candidates = [differential, singular]
    rows = []
    seen = {}
    for option in candidates:
        if option is None:
            rows.append({"witness_status": "no_linear_candidate"})
            continue
        pair = option["pair"]
        if pair not in seen:
            seen[pair] = witness(seed, pair, DIFF_CAP)
        rows.append({**option, **seen[pair], "pair": list(pair)})
    return rows


def control_figure_data(exclusions, witnesses, upper_witness):
    rows = []
    for seed, (cap, jobs) in exclusions.items():
        summary = summarize_exclusion([job.result() for job in jobs])
        if not summary["is_excluded"]:
            raise RuntimeError(f"Baseline exclusion unresolved for seed {seed}")
        differential, singular = witnesses[seed].result()
        row = {"seed": seed, "baseline_status": "excluded", "baseline_cap_checked": cap,
               "differential": differential, "sigma": singular}
        if seed == 3802:
            row["baseline_upper"] = upper_witness.result()
            if row["baseline_upper"]["witness_status"] != "certified":
                raise RuntimeError("Original-pair upper bound unresolved for seed 3802")
        rows.append(row)
    return rows[:len(PILOT_PAIRS)], rows[len(PILOT_PAIRS):]


def simulate(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        zero_run = pool.submit(zero_field_study)
        box_jobs = {3802:pool.submit(polarized_box, 3802, 201, .15),
                    3801:pool.submit(polarized_box, 3801, 151, .12)}
        face_job = pool.submit(refine_faces)
        restriction_job = pool.submit(restricted_box)
        path_audit = pool.submit(compensated_paths)
        case_jobs = [pool.submit(screen_case, family, n, seed)
                     for family, n, start, count in ensembles() + [("XXZ", 10, 2181, 4),
                                                                    ("XXZ", 12, 2201, 4)]
                     for seed in range(start, start + count)]
        geometry_jobs = [pool.submit(geometry_case, n, seed) for n, start in ((6, 2211), (8, 2231))
                         for seed in range(start, start + 4)]
        path_jobs = [pool.submit(path_case, *case) for case in (
            ("XXZ",6,1103,0), ("XXZ",8,1128,1), ("Ising-symmetric",7,1631,1),
            ("Ising-symmetric",6,1601,1))]
        seeds = (*PILOT_PAIRS, *HELD_OUT)
        witnesses = {seed:pool.submit(selected_witnesses, seed) for seed in seeds}
        upper_witness = pool.submit(witness, 3802, BASELINE, .137158)
        caps = {seed:.137 if seed == 3802 else DIFF_CAP for seed in seeds}
        exclusions = {seed:(cap, start_exclusion(pool, seed, cap)) for seed, cap in caps.items()}

        original = zero_run.result()
        dense_boxes = {seed:job.result() for seed, job in box_jobs.items()}
        coverage = [combine_coverage(dense_boxes[3801], face_job.result()),
                    combine_coverage(dense_boxes[3802])]
        restriction = restriction_job.result()
        print("Main XXZ certificates complete.", flush=True)
        if not restriction["excluded"]:
            raise RuntimeError("Restricted target exclusion failed")
        path_data, audit = path_audit.result()
        data = continuation_figure_data(case_jobs, geometry_jobs, path_jobs)
        print("Continuation calculations complete.", flush=True)
        data["pilots"], data["held_out"] = control_figure_data(exclusions, witnesses, upper_witness)
    data.update({"symmetric_grid":original.pop("grid_data"), "symmetric_restricted":original.pop("restricted_data"),
                 "compensated_path":path_data, "polarized_boxes":dense_boxes,
                 "polarized_samples":polarized_plot_samples(coverage, restriction)})
    audit = {"symmetric":original, "polarized":{"coverage":coverage, "restricted":restriction},
             "independent_tensor_checks":audit}
    return data, audit


def write_figures(data, audit, destination, file_format):
    destination.mkdir(parents=True, exist_ok=True)
    builders = [(f"Figure_{i}", fun) for i, fun in enumerate((fig1,fig2,fig3,fig4,fig5,fig6),1)]
    builders += [(f"Figure_S{i}", fun) for i, fun in enumerate((fig_s1,fig_s2,fig_s3,fig_s4),1)]
    with plt.rc_context({"font.size":9.25, "font.family":"DejaVu Sans", "axes.titlesize":9.8,
                         "axes.linewidth":.75, "legend.fontsize":7.25, "legend.frameon":False,
                         "pdf.fonttype":42, "svg.fonttype":"none"}):
        for name, build in builders:
            fig = build(data, audit)
            for axis in fig.axes:
                axis.set_title("")
            fig.savefig(destination / f"{name}.{file_format}", dpi=300, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--format", choices=("pdf", "png", "svg"), default="pdf")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    print("Running spin-chain simulations.", flush=True)
    write_figures(*simulate(args.workers), args.output, args.format)
    print(f"Figures saved in {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Development prototype: same-time cube fits with cross-view consistency gates.

This module is not part of the currently frozen online policy experiment. It
consumes visual fits only, and never chooses actions or reads simulator state.
"""
import numpy as np
from aspire_so100.geometric_state import GeometricState


def projection(fit):
    basis=np.asarray(fit['rotation'])[:,np.asarray(fit['constrained_axes'],bool)]
    return basis @ basis.T


def shared_projection(first,second):
    matrix=first @ second @ first
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    basis=vectors[:,values>np.cos(np.deg2rad(15.))**2]
    return basis @ basis.T


def quality(item):
    fit=item['fit']
    return (int(np.count_nonzero(fit['constrained_axes'])),int(fit['observed_face_count']),
            float(fit.get('inlier_fraction',0.)),-float(fit.get('surface_rms_m',1.)))


def fuse(name,observations,time):
    """Return a state or abstention, plus accepted/rejected measurement evidence.

    More constrained observations establish the same-time reference. A weaker
    view may add compatible information; it cannot overwrite conflicting observed
    directions. Conflicting equally constrained views cause abstention.
    """
    if not observations:raise ValueError('At least one same-time visual fit is required')
    ordered=sorted(observations,key=quality,reverse=True)
    state=GeometricState();accepted=[];rejected=[]
    reference_quality=quality(ordered[0])[:2]
    for item in ordered:
        fit=item['fit'];view=item['view']
        if accepted:
            prior=state.estimate(name,time);projector=projection(fit)
            common=shared_projection(prior['observed_translation_projector'],projector)
            residual=float(np.linalg.norm(common @ (np.asarray(fit['center'])-prior['center'])))
            gate=max(.006,.20*float(fit['height']))
            orientation_conflict=False
            if not prior['inferred_orientation']:
                observed_axes=np.asarray(fit['rotation'])[:,np.asarray(fit['constrained_axes'],bool)]
                alignments=np.max(np.abs(prior['rotation'].T @ observed_axes),axis=0)
                orientation_conflict=bool(np.any(alignments<np.cos(np.deg2rad(20.))))
            if residual>gate or orientation_conflict:
                evidence={'view':view,'reason':'inconsistent_same_time_cube_geometry',
                          'shared_direction_residual_m':residual,'gate_m':gate,
                          'orientation_conflict':orientation_conflict}
                rejected.append(evidence)
                if quality(item)[:2]==reference_quality:
                    return {'state':None,'accepted':[],'rejected':rejected,
                            'reason':'equally_constrained_views_disagree'}
                continue
        state.observe(name,fit,time,view);accepted.append(item)
    # Least squares over accepted measured directions; unconstrained translation
    # retains the best single-view extent estimate, without calling it observed.
    information=np.zeros((3,3));rhs=np.zeros(3);observability=np.zeros((3,3))
    for item in accepted:
        fit=item['fit'];sigma=max(.001,float(fit.get('surface_rms_m',.001)))
        weight=projection(fit)/sigma**2
        information+=weight;rhs+=weight @ np.asarray(fit['center']);observability+=projection(fit)
    # Nearly parallel measured directions must not create a spurious well-
    # localized hidden axis. Solve only in the same observable subspace used by
    # the state representation; retain extent estimates in its complement.
    eigenvalues,eigenvectors=np.linalg.eigh(observability)
    basis=eigenvectors[:,eigenvalues>.15]
    inverse=basis @ np.linalg.pinv(basis.T @ information @ basis,rcond=1e-8) @ basis.T
    anchor=np.asarray(accepted[0]['fit']['center'])
    state.objects[name]['center']=inverse @ rhs+(np.eye(3)-basis @ basis.T) @ anchor
    return {'state':state,'accepted':[item['view'] for item in accepted],'rejected':rejected,'reason':None}

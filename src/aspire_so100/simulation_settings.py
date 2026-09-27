"""Host-only, explicitly recorded contact-solver experiment settings."""


def contact_settings(value=None):
    value = {'noslip_iterations': 0} if value is None else dict(value)
    if set(value) != {'noslip_iterations'}:
        raise ValueError('Unknown recorded contact-solver settings')
    iterations = value['noslip_iterations']
    if type(iterations) is not int or not 0 <= iterations <= 100:
        raise ValueError('NoSlip iterations must be an integer in [0, 100]')
    return value


def apply_contact_settings(model, value=None):
    settings = contact_settings(value)
    model.opt.noslip_iterations = settings['noslip_iterations']
    return settings

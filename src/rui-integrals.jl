"""
    intk0(face)

Return the integral over `face` of 1/r for a field point at the center of `face`.
"""
function intk0(face)
    QA, QB, QC = face
    P = mean(face)
    QBMQA = QB - QA ; QCMQA = QC - QA ; QCMQB = QC - QB ;
	L0 = ComplexF64( 0 ) ;
    RQAP = norm( P - QA ) ;
    RQBP = norm( P - QB ) ;
    RQCP = norm( P - QC ) ;
    AR0 = SA[ RQAP, RQBP , RQCP ] ;
    ARA = SA[ RQBP, RQCP, RQAP ] ; # podría ser así : ARA = [ AR0[2], AR0[3], AR0[1] ] ;
    AOPP = SA[ norm( QBMQA ), norm( QCMQB ), norm( QCMQA ) ] ; # AOPP = [ RQAQB, RQBQC, RQCQA ] ;
    for i = 1 : 3
        R0 = AR0[ i ] ;
        RA = ARA[ i ] ;
        OPP = AOPP[ i ] ;
        if R0 < RA
            TEMP = RA ;
            RA = R0 ;
            R0 = TEMP ;
        end # END IF
        A = acos( ( RA * RA + R0 * R0 - OPP * OPP ) / 2 / RA / R0 ) ;
        B = atan( RA * sin( A ) / ( R0 - RA * cos( A ) ) ) ;
        L0 += ( R0 * sin( B ) * ( log( tan( ( B + A ) / 2 ) ) - log( tan( B / 2 ) ) ) ) ;
    end
    return L0
end

"""
    intreg(k, face;  quad=Quad7)

Return the integral over `face` of cis(k * r) / r for a field point at the center of `face`.
"""
function intreg(k, face; quad= @quad_gquts(10))
    x = mean(face)
    face0 = Face(face[1] .- x, face[2] .- x, face[3] .- x)

    # Returns (cis(k * r) - 1) / r
    function green_reg(y)
        r = norm(y)
   
            # An equivalent expression that doesn't run into indeterminations
        return im * k * cis(k / 2 * r) * sinc(k / (2π) * r)   
    end

    return (intk0(face0) + integrate_triangle(green_reg, face0; quad)) / 4π
end

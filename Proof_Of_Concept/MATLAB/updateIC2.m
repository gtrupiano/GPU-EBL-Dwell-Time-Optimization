function ICAdjusted = updateIC2(errorMatrix, ICIn)
    if(errorMatrix > 0)
        ICAdjusted = (ICIn * (1-Noise)) - 1;
    elseif(errorMatrix < 0)
        ICAdjusted = (ICIn * (1+Noise)) + 1;
    else
        ICAdjusted = ICIn;
    end

    if(ICAdjusted < 0)
        ICAdjusted = 0;
    end
end